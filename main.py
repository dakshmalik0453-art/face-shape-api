from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import mediapipe as mp
import urllib.request

app = FastAPI()
mp_face_mesh = mp.solutions.face_mesh

@app.get("/")
def home():
    return {"message": "Face Shape API is Live!"}

@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # अगर FlutterFlow ने फ़ाइल के बजाय Image URL लिंक भेजा है:
    if img is None:
        try:
            url_str = contents.decode('utf-8').strip()
            if url_str.startswith("http://") or url_str.startswith("https://"):
                req = urllib.request.urlopen(url_str)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            pass

    if img is None:
        return {"status": "error", "message": "Invalid Image or URL"}

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return {"status": "error", "message": "No face found"}

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = img.shape
        points = [{"x": int(lm.x * w), "y": int(lm.y * h)} for lm in landmarks]

        return {"status": "success", "total_points": len(points), "landmarks": points}
