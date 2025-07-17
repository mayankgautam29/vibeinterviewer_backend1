from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from docx import Document

load_dotenv()
client = OpenAI()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_docx_text(path: str):
    doc = Document(path)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

@app.post("/indexing")
async def index_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()

    if suffix not in [".pdf", ".docx"]:
        return {"message": "Only PDF and DOCX files are supported."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(file_path=tmp_path)
        docs = loader.load()
        combined_text = "\n".join(doc.page_content for doc in docs)
    else:
        combined_text = extract_docx_text(tmp_path)

    os.remove(tmp_path)

    SYSTEM_PROMPT = """
    You are a very smart resume summarizing agent. You need to summarize the text you are being given and return the brief summary of the user, including their name and key information.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined_text}
        ]
    )

    return {"message": response.choices[0].message.content}
