from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
MODEL_ID="Sourith2008/persona-chatbot"
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()