from transformers import BertConfig, BertModel

# # Building the config
# config = BertConfig()

# # Building the model from the config
# model = BertModel(config)
# # Model is randomly initialized!
# print(config)

# from transformers import BertModel
# # directly load the pretrained model
model = BertModel.from_pretrained("bert-base-cased")
# # save the model
# model.save_pretrained("test")

# to reflect the function of data transfering
sequences = ["Hello!", "Cool.", "Nice!"]
# tokenizer
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
import torch
model_inputs = torch.tensor(encoded_sequences)
output = model(model_inputs)
print(output)