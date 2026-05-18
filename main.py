from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq
import pdfplumber
import json
import io
import os
from youtube_transcript_api import YouTubeTranscriptApi

app = FastAPI()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def get_client():
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
    return Groq(api_key=GROQ_API_KEY)

app.mount("/static", StaticFiles(directory="static"), name="static")

def ask_groq(text: str, mode: str):
    client = get_client()
    
    if mode == "notes":
        prompt = f"""From this lecture text generate structured notes.
Return ONLY a JSON object, no extra text, no markdown:
{{"summary": "5 line summary here", "concepts": ["concept1", "concept2"], "definitions": [{{"term":"term1","meaning":"meaning1"}}]}}

Text: {text[:3000]}"""

    elif mode == "flashcards":
        prompt = f"""Generate 10 flashcards from this lecture text.
Return ONLY a JSON array, no extra text, no markdown:
[{{"front": "question here", "back": "answer here"}}]

Text: {text[:3000]}"""

    elif mode == "quiz":
        prompt = f"""Generate 5 multiple choice questions from this lecture text.
Return ONLY a JSON array, no extra text, no markdown:
[{{"question":"question here","options":["A) option1","B) option2","C) option3","D) option4"],"answer":"A"}}]

Text: {text[:3000]}"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a JSON generator. Always respond with valid JSON only. No explanations, no markdown, no code blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        raw = response.choices[0].message.content
        # Clean any markdown
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {str(e)} | Raw: {raw[:200]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq error: {str(e)}")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = ""
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages[:10]:  # Max 10 pages
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        notes = ask_groq(text, "notes")
        flashcards = ask_groq(text, "flashcards")
        quiz = ask_groq(text, "quiz")
        
        return {"notes": notes, "flashcards": flashcards, "quiz": quiz}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class YoutubeRequest(BaseModel):
    url: str

@app.post("/youtube")
async def process_youtube(req: YoutubeRequest):
    try:
        url = req.url
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t["text"] for t in transcript])
        
        notes = ask_groq(text, "notes")
        flashcards = ask_groq(text, "flashcards")
        quiz = ask_groq(text, "quiz")
        
        return {"notes": notes, "flashcards": flashcards, "quiz": quiz}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())
