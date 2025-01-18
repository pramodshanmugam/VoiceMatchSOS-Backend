from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from datetime import datetime

app = FastAPI()

# Directory to save uploaded audio files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are allowed.")
        
        # Save the file locally
        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        return {"message": "File uploaded successfully", "file_path": file_location}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {e}")
    

# Root route for testing
@app.get("/")
def root():
    return {"message": "FastAPI Audio Upload Service"}
