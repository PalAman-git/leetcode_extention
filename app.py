from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load GPT-J model and tokenizer
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

@app.route('/generate', methods=['POST'])
def generate():
    # Get the prompt from the request JSON body
    data = request.json
    prompt = data.get('prompt', '')

    # Tokenize the input prompt
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    # Generate text using GPT-J model
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,  # Controls randomness
        max_length=100,   # Max length of the generated text
    )

    # Decode generated tokens to text
    generated_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

    # Return the generated text as a JSON response
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    # Run the Flask app on port 80
    app.run(host='0.0.0.0', port=80)
