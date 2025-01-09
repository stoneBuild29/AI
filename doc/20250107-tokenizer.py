# tokenizer
# why I must know the concept: I cannot fix the bug which says "tokenizer"?
# the function is easy to understand: the model cannot understand the sentence, amd the funciton would convert our intexts into numerical data.
# the category: the word / the character / the subpiece (common used, to find the smallest unit that cannot seperate in the library)
# the future: maybe deal with long sequence or audio/image file, and I dont know

from transformers import BertTokenizer, AutoTokenizer
#tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# result1 = tokenizer("I have a good friend")
# print(result1)
# {'input_ids': [101, 146, 1138, 170, 1363, 1910, 102],
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1]}

# save the tokenizer
tokenizer.save_pretrained("test")

# inner steps, for understanding the dealing logic without practical use
# 1. seperate the sentence
sequence = "I have a friend." 
tokens = tokenizer.tokenize(sequence)
print(tokens)

# 2. from tokens to input IDs
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# ['I', 'have', 'a', 'friend', '.']
# [146, 1138, 170, 1910, 119]

# decoding : the reverse way
decoded_string = tokenizer.decode([146, 1138, 170, 1910, 119])
print(decoded_string)
# I have a friend.
# The transformer is formed with the encode and decode.The classical function is translating / text generation.
# the encode and decode are focusing on different functions, so some models use one of them or combine them based on goals.
<<<<<<< HEAD
# note: https://huggingface.co/learn/nlp-course/en/chapter2/4?fw=pt
=======
# note: https://huggingface.co/learn/nlp-course/en/chapter2/4?fw=pt
>>>>>>> 568f5f6 (20240107)
