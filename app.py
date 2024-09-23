import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

st.title('Disease Symptoms Generator (Fine-Tuned Model)')
st.write("This app uses your fine-tuned model to predict symptoms based on disease names.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")

# Eğitilmiş modeli yüklüyoruz
model_path = "SLM_Fine-Tuning.pt"  # Modelin yolu
model = torch.load(model_path).to(device)
model.eval()

input_str = st.text_input("Enter a disease name", "")

if st.button('Predict'):
    if input_str:
        input_ids = tokenizer.encode(input_str, return_tensors="pt").to(device)
        output = model.generate(input_ids, max_length=20, num_return_sequences=1,
                                do_sample=True, top_k=8, top_p=0.95, temperature=0.5, repetition_penalty=1.2)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        st.write(f"Model prediction: {decoded_output}")
    else:
        st.warning("Please enter a disease name!")
