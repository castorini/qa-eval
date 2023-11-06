import argparse

import torch

from FastChat.fastchat.model import load_model, get_conversation_template, add_model_args


@torch.inference_mode()
def infer_vicuna(prompt="Hello", model_name="lmsys/vicuna-13b-v1.5-16k"):
    # Load model
    model, tokenizer = load_model(
        model_name,
        device="cuda",
        num_gpus=1,
        max_gpu_memory=None,
        load_8bit=False,
        cpu_offloading=False,
        revision="main",
        debug=False,
    )

    # Build the prompt with a conversation template
    conv = get_conversation_template(model_name)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Run inference
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    temperature = 0.7
    output_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 1e-5 else False,
        temperature=temperature,
        repetition_penalty=1.0,
        max_new_tokens=512,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    # Print results
    print("-"*20)
    print(f"{conv.roles[0]}: {prompt}")
    print(f"{conv.roles[1]}: {outputs}")
    print("-"*20)
    return outputs

infer_vicuna(prompt="Hello, who are you?")