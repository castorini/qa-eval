import json
import logging
import os
import re
from typing import Optional, Sequence, Union
from tqdm import tqdm

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

from .data_utils import Candidate

logger = logging.getLogger("openllm")


CONVERSATIONAL_MODELS = {
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "HuggingFaceH4/zephyr-7b-beta",
}


def _is_conversational(model_name_or_path):
    return model_name_or_path in CONVERSATIONAL_MODELS


def _prepare(candidates, prompt_file: os.PathLike, context_file: Optional[os.PathLike] = None):
    with open(prompt_file) as p:
        prompt_template = "".join(p.readlines()).strip()

    context_passage = None
    if context_file and os.path.exists(context_file):
        with open(context_file, "r") as json_file:
            context_passage = json.load(json_file)

    prompts = []
    for candidate in candidates:
        if isinstance(candidate, Candidate):
            q = candidate.question
            gold_answers = q.answers
            candidate_answer = candidate.answer
            question = q.text
        else:
            gold_answers = candidate["answers"]
            candidate_answer = candidate["candidate_answer"]
            question = candidate["question"]

        gold_answers = ", ".join([f'"{a}"' for a in gold_answers])

        if not question.endswith("?"):
            question += "?"

        if context_passage:
            passage = context_passage.get(question, context_passage.get(question[:-1], None))
            prompt = prompt_template.format(
                q=question, answers=gold_answers, candidate_answer=candidate_answer, passage=passage
            )
        else:
            prompt = prompt_template.format(q=question, answers=gold_answers, candidate_answer=candidate_answer)

        prompts.append(prompt)
    return prompts


def _parse_response(response: str, candidate_answer: str, question: str) -> int:
    tail_matched = re.match(r".*['\"]?(yes|no)['\"]?[.!]?$", response, re.IGNORECASE)

    if response.lower().startswith("yes"):
        acceptable = "Yes"
    elif response.lower().startswith("no"):
        acceptable = "No"
    elif tail_matched:
        acceptable = tail_matched.group(1).capitalize()
    else:
        acceptable = ""
        logger.warning(f"Invalid response to `{question}` & `{candidate_answer}`: {response}")

    return int(acceptable == "Yes")


def run_inference(
    texts: Union[str, Sequence[str]],
    model,
    tokenizer,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    top_p: float = 1.0,
    num_beams: int = 1,
    batch_size: int = 1,
    num_workers: int = 16,
):
    if isinstance(texts, str):
        texts = [texts]

    model.eval()

    if _is_conversational(model.config.name_or_path):
        tokenizer.use_default_system_prompt = False
        converted_texts = []
        for t in texts:
            sections = t.split("###")
            instructions = "###".join(sections[:-1]) if len(sections) > 1 else None
            example = sections[-1].strip()

            chat = []
            if instructions:
                chat.append({"role": "system", "content": instructions})

            chat.append({"role": "user", "content": example})
            converted_texts.append(
                {"text": tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)}
            )
    else:
        converted_texts = [{"text": t} for t in texts]

    dataset = datasets.Dataset.from_list(converted_texts)

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    test_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=DataCollatorWithPadding(tokenizer, padding="longest"),
    )

    outputs = []
    for batch in tqdm(test_dataloader):
        for key in batch.keys():
            batch[key] = batch[key].to("cuda")

        input_lengths = (batch["input_ids"] != tokenizer.pad_token_id).int().sum(-1)
        batch_size = input_lengths.shape[0]

        with torch.no_grad():
            output_ids = model.generate(
                **batch,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                num_beams=num_beams,
            )

        for b in range(batch_size):
            if model.config.is_encoder_decoder:
                output_ids = output_ids[b]
            else:
                output_ids = output_ids[b, input_lengths[b] :]

            outputs.append(tokenizer.decode(output_ids, skip_special_tokens=True).strip())

    return outputs


def llm_eval(model_name_or_path: str, candidates, **kwargs):
    prompt_file = kwargs.pop("prompt_file", None)
    context_file = kwargs.pop("context_file", None)

    assert prompt_file and os.path.exists(prompt_file), "prompt_file is required in llm_eval"
    examples = _prepare(candidates, prompt_file, context_file)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    # model, tokenizer = load_model(
    #     model_name_or_path,
    #     device="cuda",
    #     num_gpus=num_gpus,
    #     max_gpu_memory=None,
    #     load_8bit=False,
    #     cpu_offloading=cpu_offloading,
    #     revision="main",
    #     debug=False,
    # )

    responses = run_inference(examples, model, tokenizer, **kwargs)

    outputs = []
    for response, candidate in zip(responses, candidates):
        acceptable = _parse_response(response, candidate.answer, candidate.question.text)
        outputs.append((acceptable, response))

    return outputs
