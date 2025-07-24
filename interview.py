from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import TypedDict
from pymongo import MongoClient
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.mongodb import MongoDBSaver
from contextlib import asynccontextmanager
import json
import os
import base64
import requests
import uuid

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["test"]
collection = db["interviews"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@tool
def save_interview(summary: str, score: int, userId: str, jobId: str):
    """
    Saves the final interview summary and numeric score to the database.
    """
    try:
        print("🔧 Saving interview...")
        collection.insert_one({
            "interviewSummary": summary,
            "interviewScore": score,
            "userId": userId,
            "jobId": jobId,
            "status": "pending"
        })
        print("✅ Saved:", summary[:60], score)
        return "Interview summary and score saved to database."
    except Exception as e:
        print("❌ Failed to save:", str(e))
        return f"Failed to save interview: {str(e)}"

class State(TypedDict):
    question: str
    answer: str
    awaiting: bool
    resume: str
    messages: list
    count: int
    audio_base64: str

tools = [save_interview]
llm = init_chat_model(model_provider="openai", model="gpt-4.1-mini")
llm_with_tools = llm.bind_tools(tools)

def get_tts_audio_base64(text: str) -> str:
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "tts-1",
            "input": text,
            "voice": "nova",
        }
    )
    return base64.b64encode(response.content).decode("utf-8")

def interview_node(state: State):
    answer = state["answer"]
    resume = state["resume"]

    SYSTEM_PROMPT = f"""
If the user has just started the interview, greet them first (only once), then proceed to ask interview questions.
Keep the interview to 5 questions only. If an answer is not given, mark score 0.
Be strict while scoring. Call `save_interview` tool only after the 5th answer and only once.

Resume Summary:
{resume}

Previous Answer:
{answer}
"""

    state["messages"].append({"role": "system", "content": SYSTEM_PROMPT})
    result = llm_with_tools.invoke(state["messages"])
    state["messages"].append({"role": "assistant", "content": result.content})
    state["question"] = result.content
    state["audio_base64"] = get_tts_audio_base64(result.content)

    if result.tool_calls:
        for tool_call in result.tool_calls:
            if tool_call["name"] == "save_interview":
                args = tool_call["args"]
                save_interview.invoke(args)

    return state

def wait_for_input_node(state: State) -> State:
    state["awaiting"] = True
    return state

def route_query(state: State) -> dict:
    if state["question"] == "":
        return {"path": "tools"}
    return {"path": "wait_for_input_node"}

tool_node = ToolNode(tools=[save_interview])

graph_builder = StateGraph(State)
graph_builder.add_node("interview_node", interview_node)
graph_builder.add_node("wait_for_input_node", wait_for_input_node)
graph_builder.add_node("route_query", route_query)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "interview_node")
graph_builder.add_edge("interview_node", "wait_for_input_node")
graph_builder.add_edge("wait_for_input_node", END)

def compile_graph_with_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

@asynccontextmanager
async def create_mongo_checkpoint(uri: str, namespace: str):
    with MongoDBSaver.from_conn_string(uri, namespace=namespace) as cp:
        yield cp

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    session_state: State = {
        "question": "",
        "answer": "",
        "awaiting": False,
        "resume": "",
        "messages": [],
        "count": 0,
        "audio_base64": "",
    }

    async with create_mongo_checkpoint(mongo_uri, "interview_sessions") as mongo_checkpoint:
        graph_with_mongo = compile_graph_with_checkpointer(mongo_checkpoint)

        while True:
            if session_state["resume"] == "":
                resume_text = await websocket.receive_text()
                session_state["resume"] = resume_text
                print("Resume received.")
                await websocket.send_text("✅ Resume received. Starting interview...")
                continue

            if session_state["count"] >= 5:
                await websocket.send_text("✅ Interview complete! Thank you for your time.")
                await websocket.close()
                break

            if session_state["question"] == "":
                state = graph_with_mongo.invoke(session_state, config=config)
                session_state.update(state)
                await websocket.send_text(json.dumps({
                    "question": state["question"],
                    "audio_base64": state["audio_base64"]
                }))
            else:
                user_input = await websocket.receive_text()
                session_state["answer"] = user_input
                session_state["awaiting"] = False
                session_state["messages"].append({"role": "user", "content": user_input})
                session_state["count"] += 1

                state = graph_with_mongo.invoke(session_state, config=config)
                session_state.update(state)
                await websocket.send_text(json.dumps({
                    "question": state["question"],
                    "audio_base64": state["audio_base64"]
                }))
