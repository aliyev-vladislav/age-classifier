import streamlit as st
from PIL import Image

import model as md

@st.cache(allow_output_mutation=True)
def init():
    return md.initialize()

def main():
    st.title("Age classifier")

    st.write("A vision transformer finetuned to classify the age of a given person's face.")
    st.write("Model source: https://huggingface.co/nateraw/vit-age-classifier")

    st.subheader("Choose a picture")
    
    picture = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=False, label_visibility="collapsed")
    col_1, col_2 = st.columns(2, gap="small")
    if picture is not None:
        image = Image.open(picture)
        with col_1:
            st.subheader("Selected image")
            st.image(image, use_column_width=True)
            button = st.button("Apply", type="primary")
        if button:
            with st.spinner("Processing..."):
                model = init()
                results = md.classify(model, image)
                with col_2:
                    st.subheader("Results")
                    for x in results:
                        st.write(x["label"], x["score"])
    st.caption("by Vladislav Aliyev")

if __name__ == "__main__":
    main()
