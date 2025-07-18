from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv
from typing import TypedDict
from typing_extensions import Annotated
from pymongo import MongoClient
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
import json
import os
import base64
import requests

load_dotenv()
app = FastAPI()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["test"]
collection = db["interviews"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@tool
def save_interview(summary: str, score: int,userId: str,jobId: str):
    """
    Saves the final interview summary and numeric score to the database.
    """
    try:
        print("🔧 Saving interview...")
        collection.insert_one({
            "interviewSummary": summary,
            "interviewScore": score,
            "userId": userId,
            "jobId": jobId
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
    audio_bytes = response.content
    return base64.b64encode(audio_bytes).decode("utf-8")

def interview_node(state: State):
    answer = state["answer"]
    resume = state["resume"]

    SYSTEM_PROMPT = f"""
If the user has just started the interview, greet them first (only once), then proceed to ask a interview questions as per the job description required. If the answer is empty just mark 0. 
If a user has already answered, skip the greeting and ask the next relevant question. Keep the interview to 5 questions only.
If the answer of the candidate is not good enough, do not mark them good. Be strict while scoring and saving the results.Do not give any response on the user's previous answer just ask them the next question thanking them for the reply. You will also get the userId and the jobId which you need to send while calling the save_interview.

Using appropriate tools, save the summary of the interview (performance of the applicant) and score of that interview out of 100.

Resume Summary (Candidate + Job Description): 
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
    print("🕒 Waiting for user input...")
    state["awaiting"] = True
    return state

def route_query(state: State) -> dict:
    print("🔀 Routing query...")
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

graph = graph_builder.compile()

paused_state = {
    "question": "",
    "answer": "",
    "awaiting": False,
    "resume": "",
    "messages": [],
    "count": 0
}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    while True:
        if paused_state["resume"] == "":
            resume_text = await websocket.receive_text()
            paused_state["resume"] = resume_text
            print(resume_text)
            await websocket.send_text("✅ Resume summary received. Starting interview...")
            continue

        if paused_state["count"] >= 5:
            await websocket.send_text("✅ Interview complete! Thank you for your time.")
            await websocket.close()
            break

        if paused_state["question"] == "":
            state = await graph.ainvoke(paused_state)
            paused_state.update(state)
            await websocket.send_text(json.dumps({
                "question": state['question'],
                "audio_base64": state['audio_base64']
            }))
        else:
            user_input = await websocket.receive_text()
            paused_state["answer"] = user_input
            paused_state["awaiting"] = False
            paused_state["messages"].append({"role": "user", "content": user_input})
            paused_state["count"] += 1

            state = await graph.ainvoke(paused_state)
            paused_state.update(state)
            await websocket.send_text(json.dumps({
                "question": state['question'],
                "audio_base64": state['audio_base64']
            }))
