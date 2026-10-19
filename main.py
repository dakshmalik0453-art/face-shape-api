import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh

def calculate_distance(p1, p2):
    """Do landmarks ke beech Euclidean distance calculate karta hai"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def analyze_face_shape(image_path_or_bytes):
    # Image load karein (OpenCV format)
    image = cv2.imread(image_path_or_bytes)
    h, w, _ = image.shape
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

        # Key Landmarks extract karein
        top_forehead = landmarks[10]
        bottom_chin = landmarks[152]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        left_jaw = landmarks[58]
        right_jaw = landmarks[288]
        left_temple = landmarks[103]
        right_temple = landmarks[332]

        # Measurements (relative distances)
        face_length = calculate_distance(top_forehead, bottom_chin)
        face_width = calculate_distance(left_cheek, right_cheek)
        jaw_width = calculate_distance(left_jaw, right_jaw)
        forehead_width = calculate_distance(left_temple, right_temple)

        # Ratios & Proportions
        ratio = face_length / face_width if face_width > 0 else 1.0
        jaw_to_cheek = jaw_width / face_width if face_width > 0 else 1.0

        # Proportions formatted string (e.g., "1 : 1.4")
        proportions_str = f"1 : {round(ratio, 1)}"

        # Jawline Type detection
        if jaw_to_cheek > 0.88:
            jawline_type = "ANGULAR"
        elif jaw_to_cheek > 0.78:
            jawline_type = "STRUCTURED"
        else:
            jawline_type = "CURVED"

        # Face Length Metric Display
        length_val = f"{round(face_length * 10, 1)}"

        # Face Shape Classification Logic
        if ratio >= 1.35 and ratio <= 1.55 and jawline_type == "CURVED":
            shape = "OVAL SHAPE"
            description = "Your face length is slightly greater than width with softly curved lines."
        elif ratio < 1.25 and jawline_type == "CURVED":
            shape = "ROUND SHAPE"
            description = "Your face width and length are almost equal with soft angles and full cheeks."
        elif ratio < 1.30 and jawline_type in ["ANGULAR", "STRUCTURED"]:
            shape = "SQUARE SHAPE"
            description = "Your face has strong, sharp jawlines with equal width and length proportions."
        elif ratio > 1.55:
            shape = "OBLONG SHAPE"
            description = "Your face is noticeably longer than it is wide with an elongated, balanced profile."
        elif forehead_width > jaw_width and jawline_type == "CURVED":
            shape = "HEART SHAPE"
            description = "You have a broader forehead that gently tapers down into a delicate chin."
        else:
            shape = "DIAMOND SHAPE"
            description = "Your cheekbones are the widest feature with a tapered forehead and jawline."

        # Final Clean JSON Structure
        return {
            "detected_shape": shape,
            "shape_description": description,
            "face_proportions": proportions_str,
            "jawline_type": jawline_type,
            "face_length": length_val
        }
