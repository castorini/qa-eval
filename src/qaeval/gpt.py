import json
import logging
import os
import re
import requests
import time
import numpy as np
from typing import Optional, Sequence, Union

import openai
from tqdm import tqdm

from .data_utils import Candidate

logger = logging.getLogger("gpt")

OPENAI_RATES = {
    "gpt-4-1106-preview": {"input": 0.01 / 1e3, "output": 0.03 / 1e3},
    "gpt-4": {"input": 0.03 / 1e3, "output": 0.06 / 1e3},
    "gpt-3.5-turbo": {"input": 0.001 / 1e3, "output": 0.002 / 1e3},
    "gpt-3.5-turbo-1106": {"input": 0.001 / 1e3, "output": 0.002 / 1e3},
}

CONVERSATIONAL_MODELS = {"gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"}


def load_model(model_name: str, **kwargs):
    azure = kwargs.pop("azure", False)
    azure_deployment_name = kwargs.pop("deployment_name", "gpt4all")
    return OpenAIProxy(model_name, azure=azure, azure_deployment_name=azure_deployment_name)


def _prepare(
    candidates,
    prompt_file: os.PathLike,
):
    with open(prompt_file) as p:
        prompt_template = "".join(p.readlines()).strip()

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
        sections = prompt.split("###")
        instruction = "###".join(sections[:-1]) if len(sections) > 1 else None
        content = sections[-1].strip()

        if instruction:
            prompts.append((content, instruction))
        else:
            prompts.append(prompt)

    return prompts


def _parse_response(response: str, candidate_answer: str, question: str) -> int:
    patterns = [
        r".*['\"]?(yes|no)\.?['\"]?[.!]?$",
        (r".*my (final )?judgment is\s+['\"]?(yes|no)['\"]?[.!]?", 2),
        r".*I would judge the candidate answer as\s+['\"]?(yes|no)['\"]?[.!]?",
        r".*\s+['\"]?(yes|no)['\"]?,? the candidate( answer)? is",
        r".*[jJ]udgment:\s+['\"]?(yes|no)\.?['\"]?",
    ]

    if response.lower().startswith("yes"):
        acceptable = "Yes"
    elif response.lower().startswith("no"):
        acceptable = "No"
    else:
        acceptable = ""
        for pattern in patterns:
            match_idx = 1
            if isinstance(pattern, (list, tuple)):
                pattern, match_idx = pattern

            matched = re.match(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)

            if matched:
                acceptable = matched.group(match_idx).capitalize()
                break

        if not acceptable:
            logger.warning(f"Invalid response to `{question}` & `{candidate_answer}`: {response}")

    return int(acceptable == "Yes")


def gpt_eval(model_name: str, candidates, **kwargs):
    prompt_file = kwargs.pop("prompt_file", None)
    assert prompt_file and os.path.exists(prompt_file), "prompt_file is required in gpt_eval"

    azure = kwargs.pop("azure", False)
    azure_deployment_name = kwargs.pop("deployment_name", "gpt4all")

    model = load_model(model_name, azure=azure, deployment_name=azure_deployment_name)
    prompts = _prepare(candidates, prompt_file)
    responses = model(prompts, **kwargs)

    outputs = []
    for response, candidate in zip(responses, candidates):
        acceptable = _parse_response(response, candidate.answer, candidate.question.text)
        outputs.append((acceptable, response))

    return outputs


class OpenAIProxy:
    def __init__(
        self,
        model_name: str,
        max_attempts: int = 5,
        sleep_time: int = 10,
        azure: bool = False,
        azure_endpoint: str = "https://dsgai-gpt4.openai.azure.com/",
        azure_api_version: str = "2023-09-01-preview",
        azure_deployment_name: str = "gpt4all",
    ):
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.sleep_time = sleep_time

        if azure:
            assert "AZURE_OPENAI_KEY" in os.environ
            openai.api_key = os.environ["AZURE_OPENAI_KEY"]
            openai.api_base = azure_endpoint
            openai.api_type = "azure"
            openai.api_version = azure_api_version
            self.deployment_name = azure_deployment_name
        else:
            assert "OPENAI_API_KEY" in os.environ
            openai.api_key = os.environ["OPENAI_API_KEY"]

        self.call_times = []
        self.num_errors = 0
        self.total_tokens = []
        self.prompt_tokens = []
        self.completion_tokens = []

    def _is_conversational(self):
        return self.model_name in CONVERSATIONAL_MODELS

    def completion(
        self,
        text: str,
        instruction: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        if self._is_conversational():
            prompt = []
            if instruction:
                prompt.append(
                    {
                        "role": "system",
                        "content": instruction,
                    }
                )
            prompt.append(
                {
                    "role": "user",
                    "content": text,
                }
            )
        else:
            if instruction:
                prompt = f"{instruction}\n\n{text}"
            else:
                prompt = text

        api_kwargs = {}
        if self.deployment_name:
            api_kwargs["engine"] = self.deployment_name

        for attempt in range(1, self.max_attempts + 1):
            try:
                s = time.time()
                if self._is_conversational():
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=prompt,
                        temperature=temperature,
                        max_tokens=max_new_tokens,
                        top_p=top_p,
                        **api_kwargs,
                    )
                else:
                    response = openai.Completion.create(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_new_tokens,
                        top_p=top_p,
                        **api_kwargs,
                    )
                e = time.time()

                self.call_times.append(e - s)
                self.total_tokens.append(response["usage"]["total_tokens"])

                if "prompt_tokens" in response["usage"] and "completion_tokens" in response["usage"]:
                    self.prompt_tokens.append(response["usage"]["prompt_tokens"])
                    self.completion_tokens.append(response["usage"]["completion_tokens"])

                if self._is_conversational():
                    return response["choices"][0]["message"]["content"].strip()
                else:
                    return response["choices"][0]["text"].strip()
            except (openai.error.ServiceUnavailableError, openai.error.RateLimitError) as e:
                logger.warning(f"[Attempt {attempt}] rate limit/availability error encountered: {e}")
                self.num_errors += 1
                time.sleep(self.sleep_time + 5 * attempt)
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[Attempt {attempt}] connection error encountered: {e}")
                self.num_errors += 1
                time.sleep(self.sleep_time + 5 * attempt)

        raise ValueError(f"API call failed after {self.max_attempts} attempts")

    def _load_cache(self, cache_file: os.PathLike):
        cache = {}
        with open(cache_file, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                cache[item["prompt"]] = item["response"]
        return cache

    def __call__(
        self,
        texts: Union[str, Sequence[str]],
        experiment_name: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        cache_dir: Optional[os.PathLike] = None,
        overwrite_cache: bool = False,
    ):
        if not cache_dir:
            cache_dir = ""

        os.makedirs(cache_dir, exist_ok=True)

        if not experiment_name:
            experiment_name = "test"

        cache_file = os.path.join(cache_dir, f"{self.model_name}_{experiment_name}.jsonl")

        cached_output = {}
        if os.path.exists(cache_file) and not overwrite_cache:
            cached_output = self._load_cache(cache_file)

        completions = []
        if isinstance(texts, str):
            completions.append(self.completion(texts))
        else:
            with open(cache_file, "w") as cw:
                for text in tqdm(texts, desc=self.model_name, colour="yellow"):
                    if isinstance(text, (list, tuple)):
                        text, instruction = text
                    else:
                        instruction = None

                    cache_key = text
                    if cache_key in cached_output:
                        response = cached_output[cache_key]
                    else:
                        response = self.completion(text, instruction, max_new_tokens, temperature, top_p)

                    cw.write(json.dumps({"prompt": text, "response": response}) + "\n")
                    cw.flush()

                    completions.append(response)

        logger.info(f"OpenAI API call stats: {self.get_stats()}")
        return completions

    def get_stats(self, reset: bool = True):
        stats = {
            "mean call time": round(np.mean(self.call_times), 3),
            "std call time": round(np.std(self.call_times), 3),
            "num calls": len(self.call_times),
            "num errors": self.num_errors,
            "total #tokens": np.sum(self.total_tokens),
            "avg #tokens": round(np.mean(self.total_tokens), 1),
        }

        if self.prompt_tokens and self.completion_tokens:
            stats["#input tokens"] = np.sum(self.prompt_tokens)
            stats["#output tokens"] = np.sum(self.completion_tokens)

            if self.model_name in OPENAI_RATES:
                rates = OPENAI_RATES[self.model_name]
                cost = np.sum(
                    [
                        p * rates["input"] + c * rates["output"]
                        for p, c in zip(self.prompt_tokens, self.completion_tokens)
                    ]
                )
                stats["cost"] = round(cost, 5)

        if reset:
            self.call_times = []
            self.num_errors = 0
            self.total_tokens = []
            self.prompt_tokens = []
            self.completion_tokens = []

        return stats
