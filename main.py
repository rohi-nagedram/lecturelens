from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
import pdfplumber
import json
import io
from youtube_transcript_api import YouTubeTranscriptApi

app = FastAPI()
client = Groq(api_key="YOUR_GROQ_API_KEY")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---- GROQ HELPER ----
def ask_groq(text: str, mode: str):
    if mode == "notes":
        prompt = f"""From this lecture text generate:
1. Summary in 5 lines
2. Key concepts as bullet points  
3. Important definitions
Return ONLY valid JSON like:
{{"summary": "...", "concepts": ["..."], "definitions": [{{"term":"...","meaning":"..."}}]}}

Text: {text[:4000]}"""

    elif mode == "flashcards":
        prompt = f"""Generate 10 flashcards from this lecture text.
Return ONLY valid JSON array like:
[{{"front": "question", "back": "answer"}}]

Text: {text[:4000]}"""

    elif mode == "quiz":
        prompt = f"""Generate 5 MCQs from this lecture text.
Return ONLY valid JSON array like:
[{{"question":"...","options":["A)...","B)...","C)...","D)..."],"answer":"A"}}]

Text: {text[:4000]}"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    raw = response.choices[0].message.content
    raw = raw.replace("```json","").replace("```","").strip()
    return json.loads(raw)

# ---- PDF ENDPOINT ----
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    text = ""
    with pdfplumber.open(io.BytesIO(contents)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    notes = ask_groq(text, "notes")
    flashcards = ask_groq(text, "flashcards")
    quiz = ask_groq(text, "quiz")
    
    return {"notes": notes, "flashcards": flashcards, "quiz": quiz}

# ---- YOUTUBE ENDPOINT ----
class YoutubeRequest(BaseModel):
    url: str

@app.post("/youtube")
async def process_youtube(req: YoutubeRequest):
    # Extract video ID
    url = req.url
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        return {"error": "Invalid YouTube URL"}
    
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([t["text"] for t in transcript])
    
    notes = ask_groq(text, "notes")
    flashcards = ask_groq(text, "flashcards")
    quiz = ask_groq(text, "quiz")
    
    return {"notes": notes, "flashcards": flashcards, "quiz": quiz}

# ---- SERVE FRONTEND ----
@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())
