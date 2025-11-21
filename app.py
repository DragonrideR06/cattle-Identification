
import streamlit as st
import torch
import faiss
import numpy as np
import os
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch.nn.functional as F

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model.eval()
    return processor, model

processor, model = load_model()


# -----------------------
# Fingerprint Embedding Function
# -----------------------
def get_fp_embedding(image):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

        feature_map = outputs.hidden_states[-1]  # (1, 2048, 7, 7)
        emb = torch.mean(feature_map, dim=[2, 3])  # GAP → (1, 2048)
        emb = F.normalize(emb, p=2, dim=1)

    return emb.squeeze().numpy()  # (2048,)


# -----------------------
# Load Database From Folder
# -----------------------
@st.cache_resource
def build_database(root_folder="dataset/"):
    database = {}
    for person in os.listdir(root_folder):
        person_path = os.path.join(root_folder, person)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            emb = get_fp_embedding(img_path)
            embeddings.append(emb)

        database[person] = np.mean(embeddings, axis=0)

    # Build FAISS index
    ids = list(database.keys())
    dim = 2048
    index = faiss.IndexFlatL2(dim)
    matrix = np.vstack([database[i] for i in ids]).astype("float32")
    index.add(matrix)

    return index, ids


index, ids = build_database("dataset/")


# -----------------------
# Identify Uploaded Fingerprint
# -----------------------
def identify_fingerprint(uploaded_img, threshold=0.81):
    emb = get_fp_embedding(uploaded_img).astype("float32")
    D, I = index.search(np.array([emb]), 1)

    matched_person = ids[I[0][0]]
    distance = D[0][0]

    # Lower distance = better match
    is_match = distance < threshold

    return matched_person, is_match


# -----------------------
# Streamlit UI
# -----------------------
st.title("🔍 Fingerprint Identification System")
st.write("Upload a fingerprint image to check if it matches any stored fingerprint.")

uploaded_file = st.file_uploader("Upload Fingerprint Image",type=["png", "jpg", "jpeg", "tif", "tiff"] )

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Fingerprint", width=250)

    if st.button("Identify"):
        with st.spinner("Comparing..."):
            img = Image.open(uploaded_file)
            person, match = identify_fingerprint(img)

        if match:
            st.success(f"✔ MATCHED — Person: {person}")
        else:
            st.error("❌ NOT MATCHED — Person Not Found")
