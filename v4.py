import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import insightface
import time
from openpyxl import load_workbook, Workbook
from openpyxl.utils.exceptions import InvalidFileException
import pickle

# === Config ===
# SUBJECT_NAME = input("Enter subject name: ").strip().title()
SAVE_INTERVAL = 10
COOLDOWN_PERIOD = timedelta(hours=1)

# === Load Face Recognition Model ===
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(1024, 1024))  # Boost detection resolution

# === Load Pretrained Embeddings ===
def load_pretrained_embeddings(embedding_path="embeddings.pkl"):
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file '{embedding_path}' does not exist... please run generate_embeddings.py first.")
    with open(embedding_path, "rb") as f:
        known_embed = pickle.load(f)
    print(f"[🔁] Loaded {len(known_embed)} embeddings from {embedding_path}")
    return known_embed

known_embeddings = load_pretrained_embeddings("embeddings.pkl")

# # === Attendance Excel Setup ===
# excel_folder = "./attendance sheets"
# os.makedirs(excel_folder, exist_ok=True)
# date_str = datetime.now().strftime("%Y-%m-%d")
# excel_path = f"{excel_folder}/attendance_{date_str}.xlsx"

# if not os.path.exists(excel_path):
#     wb = Workbook()
#     ws = wb.active
#     ws.title = SUBJECT_NAME
#     ws.append(["Name", "Time"])
#     wb.save(excel_path)
#     wb.close()
#     print(f"[📁] Created new attendance file: {excel_path}")
# else:
#     wb = load_workbook(excel_path)
#     if SUBJECT_NAME not in wb.sheetnames:
#         ws = wb.create_sheet(SUBJECT_NAME)
#         ws.append(["Name", "Time"])
#         wb.save(excel_path)
#         print(f"[📝] Added new sheet '{SUBJECT_NAME}' to {excel_path}")
#     wb.close()

# === Load Existing Attendance if any ===
attendance = {}
# try:
#     df_existing = pd.read_excel(excel_path, sheet_name=SUBJECT_NAME)
#     for _, row in df_existing.iterrows():
#         name, timestamp = row["Name"], pd.to_datetime(row["Time"])
#         attendance[name] = timestamp
# except (FileNotFoundError, InvalidFileException, ValueError):
#     pass

# === Load and Process Static Image ===
img_path = "./faces/darshan.jpg"
frame = cv2.imread(img_path)

if frame is None:
    raise FileNotFoundError(f"Could not read image from {img_path}")

# Resize for better small face detection
scale_factor = 2.0
resized_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

# Detect faces on the resized image
faces = model.get(resized_frame)

for face in faces:
    emb = face.normed_embedding
    name = "Unknown"

    for person, known_emb in known_embeddings.items():
        sim = np.dot(emb, known_emb)
        if sim > 0.6:
            name = person
            now = datetime.now()
            # print("Marked attendance for:",name)
            if name not in attendance or (now - attendance[name]) > COOLDOWN_PERIOD:
                attendance[name] = now
                print(f"[✓] {name} marked at {now.strftime('%H:%M:%S')}")
            break

    # Rescale bbox to original frame size
    x1, y1, x2, y2 = (face.bbox / scale_factor).astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# === Display the image ===
cv2.imshow("Face Attendance", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # === Save Final Attendance ===
# df_final = pd.DataFrame(
#     [(name, timestamp.strftime("%Y-%m-%d %H:%M:%S")) for name, timestamp in attendance.items()],
#     columns=["Name", "Time"]
# )
# with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
#     df_final.to_excel(writer, sheet_name=SUBJECT_NAME, index=False)
# print(f"[✔] Final attendance saved to '{excel_path}' under sheet '{SUBJECT_NAME}'.")
