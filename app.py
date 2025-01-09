from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer from Hugging Face
##Tokenizer: Converts your input text into tokens (numbers) that the model can understand.
##Model: GPT-2 is loaded from Hugging Face’s model hub.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Encode input text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
##Generation: The generate function produces text based on the given input.
##max_length: Specifies how long the generated text will be (in tokens).
##num_return_sequences: How many different sequences to generate (1 in this case).
outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
