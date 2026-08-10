from transformers import GPT2LMHeadModel,GPT2Tokenizer
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader,Dataset
from dataset import ChatDataset
from datasets import load_dataset
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name="gpt2"
model=GPT2LMHeadModel.from_pretrained(model_name)
tokenizer=GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
model.to(device)
data = load_dataset("Cynaptics/persona-chat")["train"]
data=data.select(range(1000))
dataset=ChatDataset(data,tokenizer,max_length=256)
dataloader=DataLoader(dataset,batch_size=4,shuffle=True)
epochs=5
optimizer=AdamW(model.parameters(),lr=5e-5)
model.train()
for epoch in range(epochs):
  total_loss=0
  for step,batch in enumerate(dataloader):
    input_id=batch["input_ids"].to(device)
    attention_mask=batch["attention_mask"].to(device)
    labels=batch["labels"].to(device)
    outputs=model(input_ids=input_id,attention_mask=attention_mask,labels=labels)
    loss=outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss+=loss.item()
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")