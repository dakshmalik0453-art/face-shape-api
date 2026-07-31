from fastapi import FastAPI, File, UploadFile, Form
import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import math

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

    if file is not None:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif url is not None:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req)
            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"status": "error", "message": f"Download error: {str(e)}"}
    else:
        return {"status": "error", "message": "No file or url provided"}

    if img is None:
        return {"status": "error", "message": "Invalid Image"}

    h, w, _ = img.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return {"status": "error", "message": "No face found"}

        landmarks = results.multi_face_landmarks[0].landmark
        
        # Key landmark points for face shape calculation
        # 10: Forehead/Top, 152: Chin, 234: Left Cheek, 454: Right Cheek, 172: Left Jaw, 397: Right Jaw
        p_forehead = (landmarks[10].x * w, landmarks[10].y * h)
        p_chin = (landmarks[152].x * w, landmarks[152].y * h)
        p_left_cheek = (landmarks[234].x * w, landmarks[234].y * h)
        p_right_cheek = (landmarks[454].x * w, landmarks[454].y * h)
        p_left_jaw = (landmarks[172].x * w, landmarks[172].y * h)
        p_right_jaw = (landmarks[397].x * w, landmarks[397].y * h)

        def get_dist(pt1, pt2):
            return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

        face_height = get_dist(p_forehead, p_chin)
        face_width = get_dist(p_left_cheek, p_right_cheek)
        jaw_width = get_dist(p_left_jaw, p_right_jaw)

        ratio = face_height / face_width

        # Face Shape Classification Logic
        if ratio > 1.35:
            shape = "Oblong"
        elif abs(face_height - face_width) < face_width * 0.1:
            if abs(jaw_width - face_width) < face_width * 0.1:
                shape = "Square"
            else:
                shape = "Round"
        elif jaw_width < face_width * 0.8:
            shape = "Heart / Diamond"
        else:
            shape = "Oval"

        return {
            "status": "success",
            "face_shape": shape,
            "face_height": round(face_height, 2),
            "face_width": round(face_width, 2)
        }
