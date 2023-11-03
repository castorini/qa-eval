import json
import logging
import os
from typing import Optional, Sequence, Union
from tqdm import tqdm

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from FastChat.fastchat.model import load_model

from .data_utils import Candidate

logger = logging.getLogger("openllm")


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

        prompt = prompt_template.format(q=question, answers=gold_answers, candidate_answer=candidate_answer)

        if context_passage:
            passage = context_passage[question].get("contents")
            prompt = prompt.format(passage=passage)

        prompts.append(prompt)
    return prompts


def _parse_response(response: str, candidate_answer: str, question: str) -> int:
    if response.lower().startswith("yes"):
        acceptable = "Yes"
    elif response.lower().startswith("no"):
        acceptable = "No"
    else:
        acceptable = ""
        logger.warning(f"Invalid response to `{question}` & `{candidate_answer}`: {response}")

    return int(acceptable == "Yes")


def run_inference(
    texts: Union[str, Sequence[str]],
    model,
    tokenizer,
    max_new_tokens: int = 100,
    do_sample: bool = True,
    top_p: float = 1.0,
    batch_size: int = 1,
    num_workers: int = 16,
):
    if isinstance(texts, str):
        texts = [texts]

    model.eval()

    dataset = datasets.Dataset.from_list([{"text": t} for t in texts])
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    )

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
            )

        for b in range(batch_size):
            if model.config.is_encoder_decoder:
                output_ids = output_ids[0][b]
            else:
                output_ids = output_ids[0][b, input_lengths[b] :]

            outputs.append(tokenizer.decode(output_ids, skip_special_tokens=True).strip())

    return outputs


def llm_eval(model_name_or_path: str, candidates, **kwargs):
    max_new_tokens = kwargs.pop("max_new_tokens", 100)
    num_gpus = kwargs.pop("num_gpus", 1)
    cpu_offloading = kwargs.pop("cpu_offloading", False)

    prompt_file = kwargs.pop("prompt_file", None)
    context_file = kwargs.pop("context_file", None)

    assert prompt_file and os.path.exists(prompt_file), "prompt_file is required in llm_eval"
    examples = _prepare(candidates, prompt_file, context_file)

    model, tokenizer = load_model(
        model_name_or_path,
        device="cuda",
        num_gpus=num_gpus,
        max_gpu_memory=None,
        load_8bit=False,
        cpu_offloading=cpu_offloading,
        revision="main",
        debug=False,
    )

    responses = run_inference(examples, model, max_new_tokens)

    outputs = []
    for response, candidate in zip(responses, candidates):
        acceptable = _parse_response(response, candidate.answer, candidate.question.text)
        outputs.append((acceptable, response))

    return outputs
