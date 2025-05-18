import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# Initialize the model with buffalo_l
model = FaceAnalysis(name='buffalo_l', root='.')
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
                emb = np.array(faces[0].normed_embedding)
                embeddings.append(emb)

        if embeddings:
            # Store the mean embedding for each person as a numpy array
            known_embeddings[person_name] = np.mean(embeddings, axis=0)

    with open(save_path, "wb") as f:
        pickle.dump(known_embeddings, f)
    print(f"[💾] Saved embeddings for {len(known_embeddings)} individuals to {save_path}")

register_and_save_embeddings()
