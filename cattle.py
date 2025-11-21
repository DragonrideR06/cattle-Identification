import streamlit as st
import numpy as np
import faiss
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, ResNetForImageClassification

# -------------------------------------------------------
# LOAD MODEL + PROCESSOR
# -------------------------------------------------------
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model.eval()
    return processor, model

processor, model = load_model()


# -------------------------------------------------------
# LOAD FAISS INDEX + IDS
# -------------------------------------------------------
@st.cache_resource
def load_faiss_data():
    index = faiss.read_index("faiss_index.bin")
    ids = np.load("cattle_ids.npy", allow_pickle=True).tolist()
    return index, ids

index, ids = load_faiss_data()


# -------------------------------------------------------
# EMBEDDING FOR UPLOADED IMAGE
# -------------------------------------------------------
def get_embedding_from_image(img):
    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        feat = outputs.hidden_states[-1]  # (1,2048,7,7)
        emb = torch.mean(feat, dim=[2, 3])
        emb = F.normalize(emb, p=2, dim=1)

    return emb.squeeze().numpy().astype("float32")


# -------------------------------------------------------
# IDENTIFY CATTLE
# -------------------------------------------------------
def identify_cattle(img):
    q = get_embedding_from_image(img)
    D, I = index.search(np.array([q]), 1)

    matched = ids[I[0][0]]
    distance = D[0][0]
    score = 1 / (1 + distance)

    return matched, score


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.title("🐄 Cattle Recognition System")
st.write("Upload cattle image and click Predict to identify the animal.")

uploaded_file = st.file_uploader(
    "Upload Image (JPG, PNG, TIF)",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

# Only show Predict button if file is uploaded
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Identifying cattle..."):
            cattle, score = identify_cattle(img)

        st.subheader("🔍 Result")

        # Only show matched / not matched (no score)
        if score >= 0.80:
            st.success(f"✅ Matched: {cattle}")
        else:
            st.error("❌ Not Matched")
