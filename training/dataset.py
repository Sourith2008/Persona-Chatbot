from torch.utils.data import Dataset
class ChatDataset(Dataset):
  def __init__(self,data,tokenizer,max_length):
    self.data=data
    self.tokenizer=tokenizer
    self.max_length=max_length
    self.texts=[]
    for item in data:
      dialogue=item["dialogue"]
      for i in range(0,len(dialogue)-1,2):
        user_text=dialogue[i][10:]
        bot_text=dialogue[i+1][10:]
        self.texts.append(f'User: {user_text}\n Bot: {bot_text}')
  def __len__(self):
    return len(self.texts)
  def __getitem__(self,idx):
    text=self.texts[idx]
    tokens=self.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=self.max_length,
        return_tensors="pt"
    )
    input_ids=tokens["input_ids"].squeeze(0)
    attention_mask=tokens["attention_mask"].squeeze(0)
    labels=input_ids.clone()
    return {
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "labels":labels
    }