
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import io
import torch.nn.functional as F
import nest_asyncio
import uvicorn
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Apply fix for Jupyter Notebook
nest_asyncio.apply()

app = FastAPI()

# Initialize MTCNN and FaceNet models
mtcnn = MTCNN(keep_all=True, min_face_size=40, thresholds=[0.6, 0.7, 0.7])
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def read_image(file: UploadFile):
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    
    # Convert to numpy array
    image_np = np.array(image)

    # Apply histogram equalization to enhance contrast
    img_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    image_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    return image_np

def extract_face(image):
    face_box, _ = mtcnn.detect(image)
    if face_box is None:
        raise HTTPException(status_code=400, detail="No face detected")

    x1, y1, x2, y2 = map(int, face_box[0])
    face = image[y1:y2, x1:x2]

    # Resize the face to 160x160
    face = cv2.resize(face, (160, 160))

    return face


def get_embedding(face):
    face = Image.fromarray(face).resize((160, 160))  # Resize for FaceNet
    face = np.array(face).astype(np.float32)
    face = (face - 127.5) / 128.0  # Normalize
    face = np.transpose(face, (2, 0, 1))
    face = torch.tensor(face).unsqueeze(0)

    with torch.no_grad():
        embedding = resnet(face)

    norm = torch.norm(embedding)
    if norm == 0:
        raise HTTPException(status_code=400, detail="Failed to compute face embedding")
    
    return embedding / norm  # Normalize the embedding

def cosine_similarity(embedding1, embedding2):
    """ Computes both Cosine Similarity and Euclidean Distance """
    cos_sim = F.cosine_similarity(embedding1, embedding2).item()
    euclidean_dist = torch.dist(embedding1, embedding2, p=2).item()

    # Match condition: High cosine similarity + Low Euclidean distance
    match = (cos_sim > 0.55) and (euclidean_dist < 1.0)

    return match, cos_sim  # Return both match status and score

def preprocess_image(image):
    """ Enhanced preprocessing for better OCR on Aadhaar card """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Use Bilateral Filtering to remove noise while keeping edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply Adaptive Thresholding (Better than Otsu for uneven lighting)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Use Morphological Transformations to remove small noise
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    return gray


def extract_text(image):
    """ Perform OCR with custom configurations for Aadhaar text """
    
    processed_image = preprocess_image(image)

    # Use Tesseract with a custom config
    custom_config = "--psm 6 --oem 3"  # PSM 6: Assume a uniform block of text
    text = pytesseract.image_to_string(processed_image, lang="eng", config=custom_config)
    cv2.imwrite("fixed_aadhaar.png", processed_image)  # Save for debugging

    return text


AADHAAR_REGEX = r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"  # Match 12-digit Aadhaar

def validate_aadhaar(text):
    """ Extract Aadhaar number and name from OCR text """
    print("\n🔹 Extracted OCR Text:\n", text)  # Debugging output

    # Extract Aadhaar Number (Remove extra words & keep only digits)
    aadhaar_match = re.search(AADHAAR_REGEX, text)
    aadhaar_number = aadhaar_match.group(0).replace(" ", "").replace("-", "") if aadhaar_match else "Not Found"

    # Extract Name (First capitalized words before DOB)
    name_match = re.search(r"([A-Z][a-zA-Z\s]+)\n.*DOB", text, re.MULTILINE)
    full_name = name_match.group(1).strip() if name_match else "Unknown"

    return aadhaar_number, full_name


@app.post("/verify-face")
async def verify_face(selfie: UploadFile = File(...), document: UploadFile = File(...)):
    print(f"Received selfie: {selfie.filename}, document: {document.filename}")
    
    try:
        selfie_image = read_image(selfie)
        document_image = read_image(document)

        print("Extracting faces...")
        selfie_face = extract_face(selfie_image)
        document_face = extract_face(document_image)

        print("Computing embeddings...")
        selfie_embedding = get_embedding(selfie_face)
        document_embedding = get_embedding(document_face)
        

        match, similarity_score = cosine_similarity(selfie_embedding, document_embedding)

        return JSONResponse(content={
            "match": bool(match),  # Ensure match is a boolean
            "similarity_score": float(similarity_score)  # Ensure it's a float
        })


    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/verify-face-ocr")
async def verify_face_ocr(selfie: UploadFile = File(...), document: UploadFile = File(...)):
    print(f"Received selfie: {selfie.filename}, document: {document.filename}")
    
    try:
        selfie_image = read_image(selfie)
        document_image = read_image(document)

        print("Extracting faces...")
        selfie_face = extract_face(selfie_image)
        document_face = extract_face(document_image)

        print("Computing embeddings...")
        selfie_embedding = get_embedding(selfie_face)
        document_embedding = get_embedding(document_face)
        
        match, similarity_score = cosine_similarity(selfie_embedding, document_embedding)

        # Perform OCR on the document
        print("Performing OCR on Aadhaar card...")
        text = extract_text(document_image)
        aadhaar_number, full_name = validate_aadhaar(text)

        return JSONResponse(content={
            "match": bool(match),
            "similarity_score": float(similarity_score),
            "aadhaar_number": aadhaar_number,
            "full_name": full_name
        })

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)