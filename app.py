pip install tensorflow==2.18.0
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
from fpdf import FPDF
import os


# Function to generate caption
def generate_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34, img_size=224):
    # Load the trained models and tokenizer
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Preprocess the image
    img = load_img(image_path, target_size=(img_size, img_size))  # Resize the image to 224x224
    img = img_to_array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Ensure that the image shape is compatible with the model
    if img.shape[1:] != (img_size, img_size, 3):
        raise ValueError(f"Input image must have shape ({img_size}, {img_size}, 3). Got shape {img.shape[1:]}")
    
    image_features = feature_extractor.predict(img, verbose=0)

    # Generate the caption
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


# Function to generate PDF with image and caption
def generate_pdf(image_path, caption, pdf_filename="caption.pdf"):
    # Create instance of FPDF class
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set font
    pdf.set_font("Arial", size=12)

    # Add image
    pdf.image(image_path, x=10, y=10, w=100)  # Set the image size and position

    # Add caption
    pdf.ln(85)  # Move below the image
    pdf.multi_cell(0, 10, f"Caption: {caption}")

    # Save the PDF
    pdf.output(pdf_filename)

    return pdf_filename


# Streamlit app interface
def main():
    st.set_page_config(page_title="Image Caption Generator", layout="wide")

    # Header and instructions section with styling (in Bahasa Indonesia)
    st.markdown(
        """
        <div style="text-align:center; background-color:#f0f8ff; padding:20px; border-radius:10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <h1 style="color:#1e90ff; font-family:'Arial';">🖼️ Image Caption Generator</h1>
            <p style="font-size:20px; color:#555555;">Unggah gambar dan dapatkan keterangan otomatis dalam beberapa detik!</p>
            <div style="background-color:#ffebcd; padding:15px; border-radius:8px;">
                <h3 style="color:#ff4500;">📌 Cara Menggunakan Aplikasi:</h3>
                <ol style="font-size:16px; color:#333333; text-align:left;">
                    <li><strong>Unggah Gambar:</strong> Pilih gambar dalam format JPG, JPEG, atau PNG dari perangkat Anda.</li>
                    <li><strong>Proses Gambar:</strong> Aplikasi akan memproses gambar dan menghasilkan keterangan secara otomatis.</li>
                    <li><strong>Lihat Hasil:</strong> Keterangan gambar akan ditampilkan di sebelah gambar yang Anda unggah.</li>
                </ol>
            </div>
            <p style="font-size:16px; color:#333333; margin-top:10px;">Ikuti langkah-langkah ini dan lihat bagaimana model CNN+LSTM dapat menjelaskan gambar Anda dalam kata-kata!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader("📤 Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        model_path = "model.keras"
        tokenizer_path = "tokenizer.pkl"
        feature_extractor_path = "feature_extractor.keras"

        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.image("uploaded_image.jpg", caption="Gambar yang Diunggah", use_column_width=True)

        with col2:
            with st.spinner("🔍 Menghasilkan keterangan..."):
                caption = generate_caption("uploaded_image.jpg", model_path, tokenizer_path, feature_extractor_path)

            st.markdown("### 📝 Keterangan yang Dihasilkan:")
            st.markdown(
                f"<div style='padding:10px; background-color:#f0f8ff; border-radius:10px; font-size:18px; font-weight:bold;'>"
                f"<strong>{caption}</strong></div>", unsafe_allow_html=True
            )

            st.success("✅ Keterangan berhasil dihasilkan!")

            # Dropdown untuk memilih format unduhan
            download_format = st.selectbox(
                "Pilih format unduhan:",
                ["Pilih Format", ".txt", ".pdf"]
            )

            if download_format == ".txt":
                st.download_button(
                    label="Unduh Keterangan (TXT)",
                    data=caption,
                    file_name="generated_caption.txt",
                    mime="text/plain",
                )
            elif download_format == ".pdf":
                pdf_filename = generate_pdf("uploaded_image.jpg", caption)
                st.download_button(
                    label="Unduh Keterangan dan Gambar (PDF)",
                    data=open(pdf_filename, "rb").read(),
                    file_name="generated_caption.pdf",
                    mime="application/pdf",
                )

    else:
        st.info("Silakan unggah gambar untuk memulai.")


if __name__ == "__main__":
    main()
