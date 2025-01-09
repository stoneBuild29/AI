# from transformers import pipeline

# classifier = pipeline("sentiment-analysis", device=-1)
# result1 = classifier(
#     [
#         "I have been waiting for a huggingFace course my whole life",
#         "I hate this so much",
#     ]
# )
# print(result1)

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a Hugging course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

#  inputs: {'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662,  2607,
#           2026,  2878,  2166,  1012,   102],
#         [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
#              0,     0,     0,     0,     0]]), 
#         'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])}

from transformers import AutoModel

model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

from transformers import AutoModelForSequenceClassification
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)
print(outputs.logits)

import torch 
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
result = model.config.id2label
print(result)

# The result in terminal :
# {'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662,  2607,
#           2026,  2878,  2166,  1012,   102],
#         [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
#              0,     0,     0,     0,     0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])}
# torch.Size([2, 15, 768])
# torch.Size([2, 2])
# tensor([[-2.1388,  2.1650],
#         [ 4.1692, -3.3464]], grad_fn=<AddmmBackward0>)
# tensor([[1.3336e-02, 9.8666e-01],
#         [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward0>)
# {0: 'NEGATIVE', 1: 'POSITIVE'}

# The function of pipeline: just like the model on special aims, call it and get the result 
# the steps: 1. prepare  2. model transfer 3. post handle
# The 1st step is tokenizer —— transfer input character into tensor
# The 2nd step is transfering data by pipeline, maybe a transportation ? I dont get it.
# The 3rd step is reflecting the data into the result , for example "True" or "False"