import io
import pickle
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
import PIL
# Load the pre-trained machine learning model
with open("rf_model_part_5.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define endpoint for serving index.html
@app.get("/")
async def read_index():
    with open("index.html", "r") as f:
        html_content = f.read()
    return Response(content=html_content, media_type="text/html")

# Define endpoint for image prediction
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    # Read file contents
    contents = await file.read()
    
    # Process image
    pil_image = Image.open(io.BytesIO(contents)).convert("L")
    pil_image = ImageOps.invert(pil_image)
    pil_image = pil_image.resize((28, 28), PIL.Image.LANCZOS)
    img_array = np.array(pil_image).reshape(1, -1)
    
    # Make prediction using the loaded model
    prediction = model.predict(img_array)
    
    # Return prediction result
    return {"prediction": int(prediction[0])}
