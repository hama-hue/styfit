import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

CSV_PATH = Path(__file__).resolve().parent / "clusters.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "labels.csv"

st.set_page_config(page_title="Cluster Label Editor", layout="wide")
st.title("🖼️ Clothing Label Editor")

df = pd.read_csv(CSV_PATH)

# Path where images are actually stored
IMAGES_DIR = Path("C:/Users/anish/my_website/backend/dataset/images")

if "image_path" not in df.columns:
    st.error("CSV must have an 'image_path' column")
    st.stop()

# Show table first
st.dataframe(df.head())

# Group images by existing category/occasion/body_type
group_by = st.selectbox("Group images by:", ["category", "occasion", "body_type"])
groups = df[group_by].unique()

new_labels = {}

for g in groups:
    st.markdown(f"### {group_by.capitalize()}: {g}")
    group_imgs = df[df[group_by] == g]["image_path"].tolist()
    
    cols = st.columns(5)
    for i, img_path in enumerate(group_imgs):
        img_full_path = IMAGES_DIR / Path(img_path).name
        try:
            with cols[i % 5]:
                st.image(Image.open(img_full_path), width=120, caption=Path(img_full_path).name)
        except:
            with cols[i % 5]:
                st.error(f"⚠️ {img_full_path} not found")

    new_labels[g] = st.text_input(f"New label for {g}", "")

if st.button("💾 Save Labels"):
    df[f"new_{group_by}"] = df[group_by].map(new_labels).fillna(df[group_by])
    df.to_csv(OUTPUT_CSV, index=False)
    st.success(f"✅ Labels saved to {OUTPUT_CSV}")
