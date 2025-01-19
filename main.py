from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import openai
import logging
import numpy as np
import random
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

# Reproducible results
np.random.seed(123)
random.seed(123)

# Define the model
model = DeepSpeakerModel()

# Load the checkpoint
model.m.load_weights('ResCNN_softmax_pre_training_checkpoint_102.h5', by_name=True)

# Configure OpenAI API Key
openai.api_key = "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"


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

app = FastAPI()

# Directory to save uploaded audio files
UPLOAD_DIR = "uploaded_audios"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file locally
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as audio_file:
            audio_file.write(await file.read())
        
        logging.info(f"File uploaded successfully: {file_location}")

        # Send the audio file to OpenAI Whisper API for transcription
        with open(file_location, "rb") as audio:
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio
            )

        # Extract and log the transcription text
        transcription_text = response.get("text", "")
        cleaned_transcription_text = transcription_text.replace(".", "").replace(",", "")
        
        logging.info(f"Cleaned Transcription: {cleaned_transcription_text}")

        return JSONResponse(
            status_code=200,
            content={"transcription_text": cleaned_transcription_text}
        )
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to upload or transcribe file", "error": str(e)}
        )


@app.post("/trigger/")
async def trigger(file: UploadFile = File(...)):
    try:
        # Save the uploaded file locally
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as audio_file:
            audio_file.write(await file.read())
        
        logging.info(f"File uploaded successfully: {file_location}")

        # Extract the part after the last underscore (assuming this logic is needed)
        extracted_part = file.filename.split('_')[-1]  # Extract the part after the last underscore
        file_location2 = os.path.join(UPLOAD_DIR, extracted_part)

        # Process audio files and compute similarities
        same_speaker_similarity, diff_speaker_similarity = await process_audio(file_location, file_location2)
        
        logging.info(f"Same Speaker Similarity: {same_speaker_similarity}")
        logging.info(f"Different Speaker Similarity: {diff_speaker_similarity}")

        # Check if same speaker similarity is more than 60%
        is_same_speaker = bool(same_speaker_similarity[0] >= 0.8)  # Explicitly convert to `bool`

        return JSONResponse(
            status_code=200,
            content={
                "message": "Audio processed successfully",
                "is_same_speaker": is_same_speaker,  # Will now be properly serialized
                "same_speaker_similarity": same_speaker_similarity.tolist(),
                "diff_speaker_similarity": diff_speaker_similarity.tolist()
            }
        )
    except Exception as e:
        logging.error(f"Error during audio processing: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to process file", "error": str(e)}
        )


async def process_audio(file_location1: str, file_location2: str):
    try:
        # Generate MFCC samples for the same speaker
        mfcc_001 = sample_from_mfcc(read_mfcc(file_location1, SAMPLE_RATE), NUM_FRAMES)
        mfcc_002 = sample_from_mfcc(read_mfcc(file_location2, SAMPLE_RATE), NUM_FRAMES)

        # Get embeddings for the same speaker
        predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
        predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

        # Different speaker embedding (example file used)
        mfcc_003 = sample_from_mfcc(read_mfcc('samples/1255-90413-0001.flac', SAMPLE_RATE), NUM_FRAMES)
        predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

        # Compute cosine similarities
        same_speaker_similarity = batch_cosine_similarity(predict_001, predict_002)
        diff_speaker_similarity = batch_cosine_similarity(predict_001, predict_003)

        return same_speaker_similarity, diff_speaker_similarity
    except Exception as e:
        logging.error(f"Error in process_audio: {str(e)}")
        raise
