## test the function of api key 
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Input text
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output with adjustments
outputs = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
    do_sample=True,  # Control randomness
    top_k=50,         # Limit to top 50 tokens
    top_p=0.9         # Nucleus sampling
)

# Decode and print result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
