from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

# Hugging Face API Key (replace with your token)
api_key = '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'

# Log in to Hugging Face (optional, if you are using a private model)
login(token=api_key)

def load_huggingface_model(model_name="gpt2"):
    """ Load the Hugging Face model and tokenizer """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_classifier_model(model_name="distilbert-base-uncased"):
    """ Load a classification model """
    classifier = pipeline('text-classification', model=model_name)
    return classifier

def generate_text(model, tokenizer, prompt, max_length=50):
    """ Generate text using the Hugging Face model """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

def classify_text(classifier, text):
    """ Classify text using a Hugging Face model """
    result = classifier(text)
    return result

# Example usage: Load model and generate text
if __name__ == "__main__":
    model, tokenizer = load_huggingface_model("gpt2")  # You can change model as needed
    prompt = "The future of AI is"
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"Generated Text: {generated_text}")