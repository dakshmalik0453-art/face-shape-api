from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import math

app = FastAPI()
mp_face_mesh = mp.solutions.face_mesh

@app.get("/")
def home():
    return {"message": "Face Shape & 3D Avatar API is Live!"}

def get_image_from_input(file: UploadFile = None, url: str = None):
    img = None
    if file is not None and file.filename != "":
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif url is not None and url.strip() != "":
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req)
            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            pass
    return img

# ==========================================
# FEATURE 1: Face Shape Detection (Old & Working)
# ==========================================
@app.post("/detect")
async def detect_face(file: UploadFile = File(None), url: str = Form(None)):
    img = get_image_from_input(file, url)
    if img is None:
        return {"status": "error", "message": "Invalid Image or URL"}

    h, w, _ = img.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return {"status": "error", "message": "No face found"}

        landmarks = results.multi_face_landmarks[0].landmark
        
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

        ratio = face_height / face_width if face_width > 0 else 1.0

        if ratio > 1.35:
            shape = "Oblong"
        elif abs(face_height - face_width) < face_width * 0.1:
            shape = "Square" if abs(jaw_width - face_width) < face_width * 0.1 else "Round"
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

# ==========================================
# FEATURE 2: 3D Avatar Face Texture Extractor
# ==========================================
@app.post("/extract-texture")
async def extract_texture(file: UploadFile = File(None), url: str = Form(None)):
    img = get_image_from_input(file, url)
    if img is None:
        return {"status": "error", "message": "Invalid Image or URL"}

    h, w, _ = img.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return {"status": "error", "message": "No face found"}

        landmarks = results.multi_face_landmarks[0].landmark
        
        x_coords = [int(lm.x * w) for lm in landmarks]
        y_coords = [int(lm.y * h) for lm in landmarks]

        min_x, max_x = max(0, min(x_coords)), min(w, max(x_coords))
        min_y, max_y = max(0, min(y_coords)), min(h, max(y_coords))

        box_w = max_x - min_x
        box_h = max_y - min_y
        pad = int(max(box_w, box_h) * 0.25)

        cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
        size = max(box_w, box_h) // 2 + pad

        x1 = max(0, cx - size)
        y1 = max(0, cy - size)
        x2 = min(w, cx + size)
        y2 = min(h, cy + size)

        face_crop = img[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_AREA)

        _, encoded_img = cv2.imencode('.png', face_resized)
        return Response(content=encoded_img.tobytes(), media_type="image/png")
