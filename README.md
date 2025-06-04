# 🇮🇳 Indian ChatGPT (Open Source AI Assistant)

A fully open-source, Indian-context chatbot powered by LLMs like Mistral, LLaMA, or Gemma. The goal is to create a ChatGPT-style assistant tailored for Hinglish and Indian regional needs — voice-enabled, mobile-friendly, and deployable with minimal cost.

---

## 📌 Project Structure

```bash
india-chatgpt/
├── README.md
├── data/
│   ├── raw/                # Raw Indian Hinglish/QA data
│   └── processed/          # Cleaned & formatted datasets
├── training/
│   ├── finetune_lora.py    # Script for QLoRA fine-tuning
│   └── config/             # PEFT, tokenizer, LoRA config
├── model/
│   └── base_model_loader.py
├── app/
│   ├── ui_gradio.py        # Gradio chatbot interface
│   └── ui_streamlit.py     # Streamlit version (optional)
├── voice/
│   ├── whisper_stt.py      # Speech to text
│   └── tts_playback.py     # Text to speech
├── deployment/
│   ├── streamlit_cloud/    # Hosting scripts
│   └── huggingface_space/  # Deployment config for HF Spaces
└── requirements.txt
```

---

## 🛣️ Roadmap

### ✅ Phase 1: MVP Chatbot
- [x] Load Mistral-7B or LLaMA-3 via HuggingFace
- [x] Create basic chatbot UI using Gradio/Streamlit
- [ ] Deploy basic chatbot to Hugging Face or Streamlit Cloud

### ✅ Phase 2: Data Preparation
- [ ] Scrape Hinglish or Indian conversational data
- [ ] Create Alpaca-style instruction datasets
- [ ] Format into JSONL for fine-tuning

### ✅ Phase 3: Fine-Tuning (LoRA/QLoRA)
- [ ] Use QLoRA to train model on Indian dataset
- [ ] Run on Kaggle, Colab, RunPod, or E2E cloud
- [ ] Save adapter weights for reuse

### ✅ Phase 4: Voice Integration
- [ ] Use Whisper for STT
- [ ] Add TTS with Coqui or Google TTS

### ✅ Phase 5: App & Mobile UI
- [ ] Streamlit or React frontend
- [ ] WebView Android wrapper or Kivy app

### ✅ Phase 6: RAG / Knowledge + Multilingual
- [ ] Integrate India news / UPSC data
- [ ] Add RAG pipeline (Langchain)
- [ ] Expand to Hindi, Tamil, Marathi, etc.

---

## 🧠 Model Recommendations
- [🔗 Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [🔗 LLaMA-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [🔗 Gemma 7B](https://huggingface.co/google/gemma-7b-it)

---

## 🧰 Tools and Libraries
| Task | Tool |
|------|------|
| Model Loading | HuggingFace Transformers |
| Finetuning | PEFT, TRL, Datasets |
| UI | Gradio / Streamlit |
| Voice | Whisper, Coqui TTS |
| Hosting | Hugging Face Spaces, Streamlit Cloud |

---

## 🤝 Contribute
Feel free to fork, raise issues, or collaborate. We aim to empower India with its own AI stack.

---

## 📜 License
MIT License

---

## 🙏 Credits
This project is inspired by the open-source LLM movement and the mission to bring affordable, culturally relevant AI tools to India.
