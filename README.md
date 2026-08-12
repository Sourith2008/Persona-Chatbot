# 🤖 Persona Chatbot

A conversational AI chatbot fine-tuned on the **Persona-Chat** dataset, built on top of **GPT-2** using Hugging Face `transformers`. The app is served through a **Streamlit** UI and deployed live on Streamlit Community Cloud.

**🔗 Live Demo:** [persona-chatbot-br3xg2vpjjru7qxpr5tgk5.streamlit.app](https://persona-chatbot-br3xg2vpjjru7qxpr5tgk5.streamlit.app/)

---

## 📖 Overview

Persona Chatbot generates conversational responses based on dialogue context and persona characteristics learned during fine-tuning. It's a portfolio project demonstrating an end-to-end NLP pipeline — from dataset preparation and model fine-tuning to inference and deployment.

> ⚠️ **Limitations:** As a lightweight, fine-tuned GPT-2 model, the chatbot may occasionally hallucinate, produce irrelevant responses, or lose conversational context over longer exchanges.

---

## ✨ Features

- Fine-tuned GPT-2 model trained on the [Persona-Chat](https://huggingface.co/datasets/Cynaptics/persona-chat) dataset
- Interactive chat interface built with Streamlit
- Model hosted and pulled directly from the [Hugging Face Hub](https://huggingface.co/Sourith2008/persona-chatbot)
- Simple "Clear Chat" functionality to reset conversation history

---

## 🗂️ Project Structure

```
persona-chatbot/
├── app/
│   ├── model.py             # Loads the fine-tuned model & tokenizer from Hugging Face Hub
│   └── inference.py          # Handles text generation / response logic
├── training/
│   ├── dataset.py             # Custom PyTorch Dataset for Persona-Chat
│   └── train.py                 # Fine-tuning script (GPT-2 on Persona-Chat)
├── Streamlit_ui.py            # Streamlit front-end application
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧠 Model Details

| | |
|---|---|
| **Base model** | GPT-2 |
| **Dataset** | [Cynaptics/persona-chat](https://huggingface.co/datasets/Cynaptics/persona-chat) |
| **Framework** | Hugging Face Transformers |
| **Fine-tuned weights** | [Sourith2008/persona-chatbot](https://huggingface.co/Sourith2008/persona-chatbot) |
| **Training** | 5 epochs, batch size 4, AdamW optimizer (lr=5e-5) |

The model was fine-tuned on 1,000 samples from the Persona-Chat dataset, where each conversation turn is formatted as:

```
User: <user_message>
 Bot: <bot_response>
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/persona-chatbot.git
   cd persona-chatbot
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run Streamlit_ui.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`) to start chatting.

---

## 🏋️ Training Your Own Model

If you'd like to reproduce or customize the fine-tuning process:

```bash
python training/train.py
```

This will:
1. Load `gpt2` as the base model and tokenizer
2. Load and preprocess the Persona-Chat dataset
3. Fine-tune for 5 epochs
4. Save the resulting model and tokenizer to `./model`

---

## 🛠️ Tech Stack

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Streamlit](https://streamlit.io/)

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 🙋 Author

Built by **Sourith2008** as a portfolio project exploring conversational AI fine-tuning and deployment.
