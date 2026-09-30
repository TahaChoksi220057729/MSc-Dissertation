"""
qwen3_inference.py

Loads Qwen3-8B in 4-bit quantization and provides a generation wrapper.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-8B"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    quantize_4bit: bool = True,
    compute_dtype: torch.dtype = torch.float16,
):
    """
    Loads Qwen3-8B. Requires transformers>=4.51.0 
    Requires bitsandbytes if quantize_4bit=True.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    seed: int,
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    enable_thinking: bool = False,
) -> str:
    """
    Generates a single response. `seed` is required.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    torch.manual_seed(seed)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)
