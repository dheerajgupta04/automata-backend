# import os
# import shutil
# import threading
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn

# from main import process_video_file  # we will create this function

# app = FastAPI()

# # Allow frontend access
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global state
# current_status = {
#     "processing": False,
#     "frame": 0,
#     "results": {
#         "heart_rate_bpm": None,
#         "respiratory_rate_bpm": None,
#         "spo2_percent": None,
#         "blood_pressure": None,
#         "hrv": None
#     }
# }

# UPLOAD_PATH = "temp/uploaded.mp4"


# # -----------------------------
# # Background Processing Thread
# # -----------------------------
# def background_processing():

#     global current_status

#     current_status["processing"] = True
#     current_status["frame"] = 0

#     # Always reset results properly
#     current_status["results"] = {
#         "heart_rate_bpm": None,
#         "respiratory_rate_bpm": None,
#         "spo2_percent": None,
#         "blood_pressure": None,
#         "hrv": None
#     }

#     def update_callback(frame_count, partial_results):
#         current_status["frame"] = frame_count

#         if partial_results:
#             current_status["results"].update(partial_results)

#     final_results = process_video_file(
#         UPLOAD_PATH,
#         update_callback
#     )

#     # Only update if valid
#     if final_results is not None:
#         current_status["results"].update(final_results)

#     current_status["processing"] = False


# # -----------------------------
# # Upload Endpoint
# # -----------------------------
# @app.post("/upload")
# async def upload_video(file: UploadFile = File(...)):

#     with open(UPLOAD_PATH, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     thread = threading.Thread(target=background_processing)
#     thread.start()

#     return {"message": "Processing started"}


# # -----------------------------
# # Status Endpoint
# # -----------------------------
# @app.get("/status")
# def get_status():
#     return current_status


# # -----------------------------
# # Run Server
# # -----------------------------
# if __name__ == "__main__":
#     uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from main import process_video_file

app = FastAPI()

# ---------------------------------------
# CORS (Allow frontend access)
# ---------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# Temp directory
# ---------------------------------------
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


# ---------------------------------------
# Health Check
# ---------------------------------------
@app.get("/")
def health_check():
    return {"status": "PulseScanAI API running"}


# ---------------------------------------
# Main Analyze Endpoint (SYNCHRONOUS)
# ---------------------------------------
@app.post("/api/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    age: int = Form(...),
    gender: str = Form(...),
    height: float = Form(...),
    weight: float = Form(...)
):

    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, f"{file_id}.webm")

    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Process video (blocking)
        results = process_video_file(
            video_path=file_path,
            age=age,
            gender=gender,
            height=height,
            weight=weight
        )

        if results is None:
            return JSONResponse(
                {"success": False, "error": "Not enough signal detected."},
                status_code=400
            )

        return JSONResponse({
            "success": True,
            "results": results
        })

    except Exception as e:
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )

    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)


# ---------------------------------------
# Run Server
# ---------------------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)