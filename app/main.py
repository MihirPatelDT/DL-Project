from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import os, uuid
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .processing.pipeline import process_image  # your combined logic

app = FastAPI()

origins = [
    "http://localhost:3000",  # Your local React development server
    "*"                      
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

UPLOAD_DIR = "app/uploads"
RESULT_DIR = "app/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

@app.post("/process")
async def process(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"
    input_path = os.path.join(UPLOAD_DIR, filename)

    # Save uploaded image
    with open(input_path, "wb") as f:
        f.write(await file.read())

    output_path, text = process_image(input_path)
    output_filename = os.path.basename(output_path)
    
    # Your public ngrok URL (or your final domain name)
    # IMPORTANT: Do not put a trailing slash '/' at the end
    API_BASE_URL = "https://avalyn-thermochemical-terrence.ngrok-free.dev"

    # Combine the base URL with the static path and filename
    public_image_url = f"{API_BASE_URL}/results/{output_filename}"
    return {
        "image": public_image_url,
        "text": text
    }
