from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, QuantoConfig
import torch

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_compute_dtype=torch.bfloat16
)
quantization_config = QuantoConfig(weights="int4")
DEBUG = False

def generate_response(prompt, model, tokenizer, device):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to(device)
  generated_ids = model.generate(**model_inputs, max_new_tokens=20, do_sample=True, pad_token_id=tokenizer.eos_token_id)
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
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map=device, quantization_config=quantization_config, use_cache=False)    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prompt = "Be precise, why the sky is blue?"
    response = generate_response(prompt, model, tokenizer)
    print(response)


if __name__ == "__main__":
    main()