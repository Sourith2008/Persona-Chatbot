from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID="Sourith2008/persona-chatbot"
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.to(device)
model.eval()