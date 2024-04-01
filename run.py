from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from utils import Timer
import torch

quantization_config = QuantoConfig(weights="int4")
DEBUG = True
MAX_NEW_TOKENS = 20

def generate_response(prompt, model, tokenizer, device):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to(device)
  generated_ids = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded_output = tokenizer.batch_decode(generated_ids)
  return decoded_output[0].replace(prompt, "")

def get_device(debug: bool=True) -> str:
    """Return a string representing the device where the model should live"""
    if debug: 
        device = "cpu"
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return device

def main():
    device = get_device(DEBUG)
    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    with Timer("model.from_pretrained") as timer:
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map=device, quantization_config=quantization_config, use_cache=False)    
    with Timer("tokenizer.from_pretrained") as timer:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    prompt = "Why the sky is blue?"
    with Timer("generate_response") as timer:
        response = generate_response(prompt, model, tokenizer, device)
    print(response)
    timer.print_stat()


if __name__ == "__main__":
    main()