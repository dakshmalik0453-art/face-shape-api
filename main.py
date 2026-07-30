from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from mediapipe.python.solutions import face_mesh as mp_face_mesh

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Face Shape API is Live!"}

@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "error", "message": "Invalid Image"}

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return {"status": "error", "message": "No face found"}

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = img.shape
        points = [{"x": int(lm.x * w), "y": int(lm.y * h)} for lm in landmarks]

        return {"status": "success", "total_points": len(points), "landmarks": points}
