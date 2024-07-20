from transformers import AutoTokenizer, DistilBertPreTrainedModel, DistilBertModel, DistilBertTokenizer
import torch.nn as nn
import torch
import streamlit as st
import soundfile as sf
import numpy as np
import warnings 
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import requests 
import re  
import tempfile  
import os 
import pyarrow as pa
import json 
import joblib
import re
import nltk
from nltk.corpus import words
import pickle
import sys 

# Download the words corpus if not already present
nltk.download('words', quiet=True)


with open('linreg_fluency_model.pkl', 'rb') as f:
    linreg_fluency = pickle.load(f)

with open('linreg_pronunciation_model.pkl', 'rb') as f:
    linreg_pronunciation = pickle.load(f)

print(linreg_fluency)
print(linreg_pronunciation)

# BEDI'S TALK
print(linreg_fluency.predict(np.array([[83.33 , 45.23 , 23.33]])))
print(linreg_pronunciation.predict(np.array([[83.33 , 45.23 , 23.33]])))

print("Done")

class DistilBertForRegression(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, 1)
        return logits

# Load pronunciation tokenizer
pronunciation_tokenizer = Wav2Vec2Tokenizer.from_pretrained("./models/pronunciation_tokenizer")

# Load pronunciation model
pronunciation_model = Wav2Vec2ForCTC.from_pretrained("./models/pronunciation_model")

# Load fluency tokenizer
fluency_tokenizer = DistilBertTokenizer.from_pretrained("./models/fluency_tokenizer")

# Load fluency model
fluency_model = DistilBertForRegression.from_pretrained("./models/fluency_model")

print("All models and tokenizers loaded successfully.")


def count_misspelled_words(text):
    english_words = set(words.words())
    words_in_text = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words_in_text)
    misspelled = [word for word in words_in_text if word not in english_words]

    incorrect_count = len(misspelled)

    return f"{(incorrect_count / total_words * 100):.2f}"


def get_fluency_score(transcription):
    tokenized_text = fluency_tokenizer(transcription, return_tensors="pt")
    with torch.no_grad():
        output = fluency_model(**tokenized_text)
    fluency_score = output.item()
    return round(fluency_score, 2)

def download_word_list():
    print("Downloading English word list...")
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    response = requests.get(url)
    words = set(response.text.split())
    print("Word list downloaded.")
    return words

english_words = download_word_list()

# Function to count correctly spelled words in text
def count_spelled_words(text, word_list):
    print("Counting spelled words...")
    # Split the text into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    correct = sum(1 for word in words if word in word_list)
    incorrect = len(words) - correct
    
    print("Spelling check complete.")
    return incorrect, correct

# Function to apply spell check to an item (assuming it's a dictionary)
def apply_spell_check(item, word_list):
    print("Applying spell check...")
    if isinstance(item, dict):
        # This is a single item
        text = item['transcription']
        incorrect, correct = count_spelled_words(text, word_list)
        item['incorrect_words'] = incorrect
        item['correct_words'] = correct
        print("Spell check applied to single item.")
        return item
    else:
        # This is likely a batch
        texts = item['transcription']
        results = [count_spelled_words(text, word_list) for text in texts]
        
        incorrect_counts, correct_counts = zip(*results)
        
        item = item.append_column('incorrect_words', pa.array(incorrect_counts))
        item = item.append_column('correct_words', pa.array(correct_counts))
        
        print("Spell check applied to batch of items.")
        return item


def get_pronunciation_score(transcription, progress_bar, status_area):
    progress_bar.progress(0)
    status_area.text("Starting pronunciation scoring...")
    
    incorrect, correct = count_spelled_words(transcription, english_words)
    progress_bar.progress(33)
    status_area.text(f"Spelling check - Incorrect words: {incorrect}, Correct words: {correct}")
    
    # Calculate pronunciation score
    fraction = correct / (incorrect + correct)
    score = round(fraction * 100, 2)
    progress_bar.progress(66)
    status_area.text(f"Pronunciation score for '{transcription}': {score}")
    
    progress_bar.progress(100)
    status_area.text("Pronunciation scoring process complete.")
    
    return {
        "transcription": transcription,
        "pronunciation_score": score
    }

def get_pronunciation_and_fluency_scores(transcription, progress_bar, status_area):
    progress_bar.progress(0)
    status_area.text("Starting pronunciation and fluency scoring...")
    
    incorrect, correct = count_spelled_words(transcription, english_words)
    progress_bar.progress(25)
    status_area.text(f"Spelling check - Incorrect words: {incorrect}, Correct words: {correct}")
    
    # Calculate pronunciation score
    fraction = correct / (incorrect + correct)
    pronunciation_score = round(fraction * 100, 2)
    progress_bar.progress(50)
    status_area.text(f"Pronunciation score calculated: {pronunciation_score}")
    
    # Calculate fluency score
    fluency_score = get_fluency_score(transcription)
    progress_bar.progress(75)
    status_area.text(f"Fluency score calculated: {fluency_score}")
    
    progress_bar.progress(100)
    status_area.text("Pronunciation and fluency scoring process complete.")
    
    return {
        "transcription": transcription,
        "pronunciation_score": pronunciation_score,
        "fluency_score": fluency_score
    }

def transcribe_audio(audio_path, progress_bar, status_area):
    progress_bar.progress(0)
    status_area.text("Starting audio transcription...")
    
    warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
    warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")
    
    # Load audio file
    audio, sample_rate = sf.read(audio_path)
    progress_bar.progress(25)
    status_area.text("Audio file loaded successfully.")
    
    # Check if the audio is mono
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Resample if needed (Wav2Vec2 expects 16kHz)
    if sample_rate != 16000:
        # Simple resampling (less accurate but doesn't require librosa)
        audio = np.array(audio[::int(sample_rate/16000)])
        status_area.text("Audio resampled to 16kHz.")
    
    input_values = pronunciation_tokenizer(audio, return_tensors = "pt").input_values
    progress_bar.progress(50)
    status_area.text("Audio tokenized.")
    
    logits = pronunciation_model(input_values).logits
    progress_bar.progress(75)
    status_area.text("Model inference complete.")
    
    prediction = torch.argmax(logits, dim = -1)
    transcription = pronunciation_tokenizer.batch_decode(prediction)[0]
    
    progress_bar.progress(100)
    status_area.text(f"Transcription complete: {transcription.lower()}")
    
    return transcription.lower()

st.set_page_config(page_title="Speech Pronunciation Scorer", layout="wide")

st.title("🎙️ Speech Pronunciation Scorer")

st.write("""
Upload your speech audio file to get a transcription and pronunciation score.
This tool uses advanced speech recognition to transcribe your audio and evaluate pronunciation quality.
""")

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Transcribe and Score"):
        progress_bar = st.progress(0)
        status_area = st.empty()
        
        with st.spinner("Processing audio..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Transcribe audio
            transcription = transcribe_audio(tmp_file_path, progress_bar, status_area)
            
            # Reset progress for next step
            progress_bar.empty()
            
            # Get pronunciation and fluency scores
            result = get_pronunciation_and_fluency_scores(transcription, progress_bar, status_area)

            base_pronunciation_score = result["pronunciation_score"]
            base_fluency_score = result["fluency_score"]
            incorrect_words_percentage = count_misspelled_words(transcription)

            print("Base Pronunciation Score:" , base_pronunciation_score)
            print("Base fluency Score:" , base_fluency_score)
            
            final_pronunciation_score = pronunciation_model.predict(np.array([[base_pronunciation_score , base_fluency_score , incorrect_words_percentage]]))
            final_fluency_score = fluency_model.predict(np.array([[base_pronunciation_score , base_fluency_score , incorrect_words_percentage]]))
            
            # Remove temporary file
            os.unlink(tmp_file_path)

        # Clear the progress bar and status area
        progress_bar.empty()
        status_area.empty()

        # Display results
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Transcription")
            st.write(result['transcription'])
        
        with col2:
            st.markdown("### Pronunciation Score")
            st.metric(label="Score", value=f"{final_pronunciation_score}%")
        
        with col3:
            st.markdown("### Fluency Score")
            st.metric(label="Score", value=f"{final_fluency_score}")

        # Display JSON
        st.subheader("JSON Output")
        st.json(json.dumps(result, indent=2))
        
st.markdown("---")