from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import pandas as pd
import os
import shutil

app = FastAPI()

model = YOLO("yolov8n.pt")
pose = mp.solutions.pose.Pose()

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_angle_change(a, b):
    return abs(np.arctan2(b[1]-a[1], b[0]-a[0]))

@app.post("/predict_touches")
async def predict_touches(video: UploadFile = File(...), player_name: str = "Unknown"):
    temp_path = f"temp_{video.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    cap = cv2.VideoCapture(temp_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(3)), int(cap.get(4))

    touch_count = 0
    skill_score = 0
    ball_positions = []
    direction_changes = 0
    min_touch_distance = 35
    stability_scores = []
    show_touch_effect = 0

    out = cv2.VideoWriter("scout_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        pose_result = pose.process(rgb)

        ball_center = None
        player_foot = None
        touched = False

        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cls = int(det.cls[0])
            if cls == 32:
                ball_center = ((x1 + x2)//2, (y1 + y2)//2)

        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark
            foot = lm[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX]
            hip = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
            fx, fy = int(foot.x * w), int(foot.y * h)
            hx, hy = int(hip.x * w), int(hip.y * h)
            player_foot = (fx, fy)
            movement = abs(hip.y - foot.y)
            stability = max(0, 1 - movement * 3)
            stability_scores.append(stability)

        if ball_center and player_foot:
            dist = get_distance(ball_center, player_foot)
            if dist < min_touch_distance:
                touch_count += 1
                touched = True
                show_touch_effect = 5

        if ball_center:
            ball_positions.append(ball_center)
            if len(ball_positions) >= 6:
                a, b = ball_positions[-6], ball_positions[-1]
                angle = get_angle_change(a, b)
                if 'last_angle' in locals() and abs(angle - last_angle) > 0.6:
                    direction_changes += 1
                    skill_score += 1
                last_angle = angle

        out.write(frame)

    cap.release()
    out.release()
    os.remove(temp_path)

    avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0

    level = "متوسط"
    if touch_count >= 12 and skill_score >= 4 and avg_stability > 0.7:
        level = "احترافي"
    elif touch_count >= 8 and avg_stability > 0.5:
        level = "مستوى عالي"

    return JSONResponse(content={
        "Name": player_name,
        "Touches": touch_count,
        "Skill Score": skill_score,
        "Avg Stability": round(avg_stability, 2),
        "Level": level
    })
