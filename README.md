# Scheme Research Tool (Free/Open-Source Version)

## 🚀 Overview
This Streamlit app processes PDF URLs of government schemes, builds a FAISS index using free sentence-transformer embeddings, and lets users ask questions using a lightweight QA model.

## 🧩 Features
- Load and process PDF URLs
- Embeds using all-MiniLM-L6-v2 (free)
- QA using HuggingFace's DistilBERT (no OpenAI key needed)

## ⚙️ Requirements
```bash
pip install -r requirements.txt
```

## 🖥️ Run the App
```bash
streamlit run main.py
```

## 🔧 Notes
- No OpenAI API needed
- You can update models to larger ones if GPU is available


## 📂 File Structure
- `main.py`: Streamlit frontend
- `utils.py`: Core processing logic
- `requirements.txt`: Dependencies
- `.config`: Secret API Key (not pushed)
- `faiss_store_openai.pkl`: FAISS index
