import torch 
from app.model import model,tokenizer,device
def infer(user_input):
    prompt=f'User: {user_input}\n Bot:'
    inputs=tokenizer(prompt,return_tensors='pt')
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }
    outputs=model.generate(
      **inputs,
      do_sample=True,
      top_k=5,
      top_p=0.9,
      temperature=0.7,
      repetition_penalty=1.2,
      max_length=50
      )
    input_length=inputs['input_ids'].shape[1]
    response=tokenizer.decode(outputs[0][input_length:],skip_special_tokens=True)
    response=response.strip()
    return response
