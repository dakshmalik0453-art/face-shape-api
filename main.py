from fastapi import FastAPI, File, UploadFile, Form
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
async def detect_face(
    file: UploadFile = File(None), 
    url: str = Form(None)
):
    img = None

    # 1. अगर FlutterFlow ने असली File भेजी है:
    if file is not None:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. अगर FlutterFlow ने Image का URL भेजा है:
    elif url is not None:
        try:
            # URL से फोटो डाउनलोड करने का कोड
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req)
            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"status": "error", "message": f"URL से फोटो डाउनलोड नहीं हो पाई: {str(e)}"}
    
    # 3. अगर दोनों ही नहीं भेजे:
    else:
        return {"status": "error", "message": "कृपया 'file' या 'url' में से कोई एक भेजें"}

    # अगर इमेज प्रोसेस नहीं हो पाई
    if img is None:
        return {"status": "error", "message": "Invalid Image"}

    # MediaPipe से चेहरा ढूँढना
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return {"status": "error", "message": "फोटो में कोई चेहरा नहीं मिला"}

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = img.shape
        points = [{"x": int(lm.x * w), "y": int(lm.y * h)} for lm in landmarks]

        return {"status": "success", "total_points": len(points), "landmarks": points}
