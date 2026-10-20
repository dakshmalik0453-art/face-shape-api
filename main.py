from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import mediapipe as mp
import math
import urllib.request

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_face_mesh = mp.solutions.face_mesh

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# FlutterFlow se aane wale JSON body ke liye schema
class FaceAnalysisRequest(BaseModel):
    image_url: str

@app.get("/")
def root():
    return {"status": "Face Analysis API is running!"}

@app.post("/analyze-face")
async def analyze_face(req: FaceAnalysisRequest):
    try:
        # URL se image download karein
        req_headers = {"User-Agent": "Mozilla/5.0"}
        request_obj = urllib.request.Request(req.image_url, headers=req_headers)
        with urllib.request.urlopen(request_obj) as response:
            image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return {"error": f"Failed to download image from URL: {str(e)}"}

    if image is None:
        return {"error": "Invalid image format"}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return {"error": "No face detected"}

        landmarks = results.multi_face_landmarks[0].landmark

        top_forehead = landmarks[10]
        bottom_chin = landmarks[152]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        left_jaw = landmarks[58]
        right_jaw = landmarks[288]
        left_temple = landmarks[103]
        right_temple = landmarks[332]

        face_length = calculate_distance(top_forehead, bottom_chin)
        face_width = calculate_distance(left_cheek, right_cheek)
        jaw_width = calculate_distance(left_jaw, right_jaw)
        forehead_width = calculate_distance(left_temple, right_temple)

        ratio = face_length / face_width if face_width > 0 else 1.0
        jaw_to_cheek = jaw_width / face_width if face_width > 0 else 1.0

        proportions_str = f"1 : {round(ratio, 1)}"

        if jaw_to_cheek > 0.88:
            jawline_type = "ANGULAR"
        elif jaw_to_cheek > 0.78:
            jawline_type = "STRUCTURED"
        else:
            jawline_type = "CURVED"

        length_val = f"{round(face_length * 10, 1)}"

        if 1.35 <= ratio <= 1.55 and jawline_type == "CURVED":
            shape = "OVAL SHAPE"
            desc = "Your face length is slightly greater than width with softly curved lines"
        elif ratio < 1.25 and jawline_type == "CURVED":
            shape = "ROUND SHAPE"
            desc = "Your face width and length are almost equal with soft angles."
        elif ratio < 1.30 and jawline_type in ["ANGULAR", "STRUCTURED"]:
            shape = "SQUARE SHAPE"
            desc = "Your face has strong, sharp jawlines with balanced proportions."
        elif ratio > 1.55:
            shape = "OBLONG SHAPE"
            desc = "Your face is noticeably longer than it is wide."
        elif forehead_width > jaw_width and jawline_type == "CURVED":
            shape = "HEART SHAPE"
            desc = "Forehead is slightly broader, tapering into a delicate chin."
        else:
            shape = "DIAMOND SHAPE"
            desc = "Cheekbones are wide with a tapered forehead and jawline."

        return {
            "detected_shape": shape,
            "shape_description": desc,
            "face_proportions": proportions_str,
            "jawline_type": jawline_type,
            "face_length": length_val
        }
