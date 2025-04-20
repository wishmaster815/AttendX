import os
import cv2
import numpy as np
import pickle
import insightface
from insightface.app import FaceAnalysis

# In your generate_embeddings.py
model = FaceAnalysis(name='antelopev2')
model.prepare(ctx_id=0, det_size=(1280, 1280))


def register_and_save_embeddings(dataset_path="faces", save_path="embeddings.pkl"):
    known_embeddings = {}
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        embeddings = []
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = model.get(img)
            if faces:
                emb = faces[0].normed_embedding
                embeddings.append(emb)

        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            known_embeddings[person_name] = avg_emb

    with open(save_path, "wb") as f:
        pickle.dump(known_embeddings, f)
    print(f"[💾] Saved {len(known_embeddings)} embeddings to {save_path}")

register_and_save_embeddings()
