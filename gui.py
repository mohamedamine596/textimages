import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

API_URL = "http://localhost:8000/generate-image"  # local FastAPI

st.set_page_config(page_title="Text to Image Generator", layout="centered")

st.title("🎨 Text to Image Generator")
st.markdown("Generate images using Stable Diffusion hosted on Google Colab (via local API proxy).")

with st.form("image_form"):
    prompt = st.text_input("Enter prompt", "a futuristic city at dawn")
    width = st.slider("Width", 256, 1024, 512, step=64)
    height = st.slider("Height", 256, 1024, 512, step=64)
    seed = st.number_input("Seed (optional)", min_value=0, value=0)
    use_seed = st.checkbox("Use this seed", value=False)
    
    submitted = st.form_submit_button("Generate Image")

if submitted:
    st.info("Generating image... Please wait ⏳")

    payload = {
        "prompt": prompt,
        "width": width,
        "height": height
    }
    if use_seed:
        payload["seed"] = seed

    try:
        res = requests.post(API_URL, json=payload)
        res.raise_for_status()
        result = res.json()

        img_path = result["filename"]
        st.success("Image generated successfully!")
        
        # Show image
        with open(img_path, "rb") as img_file:
            img_bytes = img_file.read()
            image = Image.open(BytesIO(img_bytes))
            st.image(image, caption=f"Prompt: {prompt}", use_column_width=True)

            # Download button
            b64 = base64.b64encode(img_bytes).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="{img_path.split("/")[-1]}">📥 Download Image</a>'
            st.markdown(href, unsafe_allow_html=True)

    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
