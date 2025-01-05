from transformers import pipeline


# model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
# pipe = pipeline("sentiment-analysis", model=model_name, device=-1)  # CPU
# # test the words
# result = pipe(["I have been waiting for a HuggingFace course my whole life", "I hate this so much!"])
# print(result)

# pipe1 = pipeline("zero-shot-classification", model=model_name, device=-1)
# result1 = pipe1("This is a coding exercise", candidate_labels = ["education", "politics", "business"])
# print(result1)

# model_name1 = "gpt2"
# pipe2 = pipeline("text-generation", model=model_name1, device=-1)
# result2 = pipe2("In this exercise, I will ", max_length=50, truncation=True)
# print(result2)

# generator = pipeline("text-generation", model="distilgpt2", device=-1)
# result3 = generator(
#     "In this course, we will teach you how to",
#     max_length=30,
#     num_return_sequences=2,
#     truncation=True
# )
# print(result3)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en", device=-1) #Helsinki-NLP/opus-mt-fr-en
result4 = translator("Ce cours est produit par Hugging Face.")
print(result4)