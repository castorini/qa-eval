import json
import logging
import os
import requests
import time
import numpy as np
from typing import Optional, Sequence, Union

import openai

logger = logging.getLogger("gpt")


def load_model(model_name: str, **kwargs):
    max_tokens = kwargs.pop("max_tokens", 100)
    temperature = kwargs.pop("temperature", 0.0)
    return OpenAIProxy(model_name, max_tokens, temperature)


def _prepare(candidates):
    for candidate in candidates:
        answers = ", ".join([f'"{a}"' for a in candidate.question.answers])

        q = candidate.question.text
        if not q.endswith("?"):
            q += "?"

        prompt = f"Question: {q}\nAnswer: {answers}\nCandidate: {candidate.answer}\n\nIs candidate correct?"
        yield prompt


def _parse_response(response: str, candidate_answer: str, question: str) -> int:
    if response.lower().startswith("yes"):
        acceptable = "Yes"
    elif response.lower().startswith("no"):
        acceptable = "No"
    else:
        acceptable = ""
        logger.warning(f"Invalid response to `{question}` & `{candidate_answer}`: {response}")

    return int(acceptable == "Yes")


def gpt_eval(model_name: str, candidates, **kwargs):
    model = load_model(model_name, **kwargs)
    prompts = list(_prepare(candidates))
    responses = model(prompts, "test", **kwargs)

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
    ):
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.sleep_time = sleep_time

        assert "OPENAI_API_KEY" in os.environ
        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.call_times = []
        self.num_errors = 0
        self.total_tokens = []

    def _is_conversational(self):
        return self.model_name in ("gpt-4", "gpt-3.5-turbo")

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
                    )
                else:
                    response = openai.Completion.create(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_new_tokens,
                        top_p=top_p,
                    )
                e = time.time()

                self.call_times.append(e - s)
                self.total_tokens.append(response["usage"]["total_tokens"])
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
                for text in texts:
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
        num_calls = len(self.call_times)
        avg_call_time = np.mean(self.call_times)
        std_call_time = np.std(self.call_times)

        num_errors = self.num_errors

        total_tokens = np.sum(self.total_tokens)
        avg_tokens = np.mean(self.total_tokens)

        if reset:
            self.call_times = []
            self.num_errors = 0
            self.total_tokens = []

        return {
            "mean call time": avg_call_time,
            "std call time": std_call_time,
            "num calls": num_calls,
            "num errors": num_errors,
            "total #tokens": total_tokens,
            "avg #tokens": avg_tokens,
        }
