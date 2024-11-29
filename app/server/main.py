from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from model.inference import inference  # Import the inference function
import io

app = FastAPI()

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, runs it through the model, and returns the predicted class.
    """
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="File must be a JPG or PNG image.")

        # Read and convert the uploaded file to a PIL image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Perform inference
        predicted_class = inference(image)

        return {"filename": file.filename, "predicted_class": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

