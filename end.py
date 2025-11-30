import streamlit as st
from deep_translator import GoogleTranslator
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# ----------------------------
# MODEL PATH
# ----------------------------
MODEL_PATH = r"C:\Users\gold\Documents\python\final project\flan_t5_small_model"

# ----------------------------
# LOAD TOKENIZER & MODEL
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float32
)

model.to("cpu")

# ----------------------------
# CLEAN TEXT
# ----------------------------
def clean_repeats(text):
    return re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text)

# ----------------------------
# CORE MODEL FUNCTION
# ----------------------------
def answer_medical_question(question, focus="General Health"):
    input_text = f"Question about {focus}: {question}"

    inputs = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=150,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=2.5,
        early_stopping=True
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_repeats(raw)

# ----------------------------
# MULTILINGUAL HANDLER
# ----------------------------
def multilingual_answer(user_msg, focus="General Health"):
    try:
        lang = detect(user_msg)
    except:
        lang = "en"

    if lang != "en":
        english_query = GoogleTranslator(source='auto', target='en').translate(user_msg)
    else:
        english_query = user_msg

    reply_en = answer_medical_question(english_query, focus)

    if lang != "en":
        reply = GoogleTranslator(source='en', target=lang).translate(reply_en)
        disclaimer = GoogleTranslator(source='en', target=lang).translate(
            "This is general medical information, not a substitute for professional diagnosis."
        )
    else:
        reply = reply_en
        disclaimer = "This is general medical information, not a substitute for professional diagnosis."

    return reply + "\n\n" + disclaimer

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Multilingual Medical Chatbot", page_icon="🩺")
st.title("🩺 Multilingual Medical Support Chatbot")

user_question = st.text_area("Type your medical question:", height=140)
focus_area = st.text_input("Focus area (optional):", value="General Health")

if st.button("Get Answer"):
    if user_question.strip():
        with st.spinner("Generating answer..."):
            output = multilingual_answer(user_question, focus_area)
        st.success("Response:")
        st.write(output)
    else:
        st.error("Please type a question first.")

