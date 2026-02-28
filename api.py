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
    height: float = Form(None),
    weight: float = Form(None)
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
        # Clean up temp file (TEMPORARILY DISABLED FOR DEBUGGING)
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        pass


# ---------------------------------------
# Run Server
# ---------------------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=3000, reload=True)