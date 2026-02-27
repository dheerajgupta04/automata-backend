import os
import shutil
import threading
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from main import process_video_file  # we will create this function

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
current_status = {
    "processing": False,
    "frame": 0,
    "results": {
        "heart_rate_bpm": None,
        "respiratory_rate_bpm": None,
        "spo2_percent": None,
        "blood_pressure": None,
        "hrv": None
    }
}

UPLOAD_PATH = "temp/uploaded.mp4"


# -----------------------------
# Background Processing Thread
# -----------------------------
def background_processing():

    global current_status

    current_status["processing"] = True
    current_status["frame"] = 0

    # Always reset results properly
    current_status["results"] = {
        "heart_rate_bpm": None,
        "respiratory_rate_bpm": None,
        "spo2_percent": None,
        "blood_pressure": None,
        "hrv": None
    }

    def update_callback(frame_count, partial_results):
        current_status["frame"] = frame_count

        if partial_results:
            current_status["results"].update(partial_results)

    final_results = process_video_file(
        UPLOAD_PATH,
        update_callback
    )

    # Only update if valid
    if final_results is not None:
        current_status["results"].update(final_results)

    current_status["processing"] = False


# -----------------------------
# Upload Endpoint
# -----------------------------
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):

    with open(UPLOAD_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    thread = threading.Thread(target=background_processing)
    thread.start()

    return {"message": "Processing started"}


# -----------------------------
# Status Endpoint
# -----------------------------
@app.get("/status")
def get_status():
    return current_status


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)