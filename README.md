# Scheme Research Tool

## 🚀 Overview
This Streamlit web app loads URLs (pointing to PDFs of government schemes), processes and summarizes them using OpenAI embeddings, and allows users to query the schemes interactively.

## 🧩 Features
- Load and process scheme URLs
- Summarize scheme info
- Ask questions and get contextual answers

## ⚙️ Requirements
```bash
pip install -r requirements.txt
```

## 🗝️ API Key Setup
Create a `.config` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 🖥️ Running the App
```bash
streamlit run main.py
```

## 📂 File Structure
- `main.py`: Streamlit frontend
- `utils.py`: Core processing logic
- `requirements.txt`: Dependencies
- `.config`: Secret API Key (not pushed)
- `faiss_store_openai.pkl`: FAISS index
