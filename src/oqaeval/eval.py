"""
*** This script calls OpenAI APIs that will charge you per use

OPENAI_API_KEY env variable should be set to run this script
"""
import csv
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

from .data_utils import read_questions, read_predict_file, read_annotations, Question, Candidate
from .gpt import gpt_eval
from .openllm import llm_eval
from .squad_evaluate import metric_max_over_ground_truths, regex_match, exact_match_score, f1_score
# from .vicuna_llm import infer_vicuna

logger = logging.getLogger("eval")

OPENAI_MODELS = (
    "text-davinci-003",
    "gpt-3.5-turbo",
    "gpt-4",
)


def _is_openai_model(model_name: str) -> bool:
    return model_name in OPENAI_MODELS


# def vicuna_eval(question: Question, candidate_answer: str) -> Tuple[int, str]:
#     answers = " or ".join(question.answers)
#     q = question.text
#
#     with open("../passage_dpr.json", "r") as json_file:
#         context_passage = json.load(json_file)
#
#     passage = context_passage.get(q, {}).get("contents")
#
#     if not q.endswith("?"):
#         q += "?"
#
#     prompt = f"""
#         You are an expert judge of a content. You'll be give a question, ground-truth answers
#         and a candidate answer. Using your inbuilt knowledge, given context and given ground-truth answers try to
#         verify if the candidate is correct or not. Provide explanation for the comparison and provide
#         answer based on the explanation as "yes" or "no". Here, "yes" represents that the candidate answer
#         is relevant and correct based on either inbuilt knowledge, given context or given ground-truth answers. If not,
#         the answer based on the explanation would be "no".
#
#         Here are some of the sample examples:
#
#         Question: how long have we been celebrating pi day
#         Ground-Truth Answers: "1988", "2009"
#         Candidate: We have been celebrating pi day since 1988.
#
#         Is the candidate correct?
#         Since 1998 we have been celebrating pi day. Answer based on explanation: yes.
#
#         ###
#         Question: who plays dylan in daddy's home 2
#         Ground-Truth Answers: "Owen Vaccaro"
#         Candidate: Vaccaro
#
#         Is the candidate correct?
#         Owen Vaccaro plays Dylan in the movie "Daddy's Home 2". Vaccaro is the last name. Answer based on explanation: yes.
#
#         ###
#         Now make prediction for following data.
#         Compulsory format for the `Is the candidate correct?` question's answer:
#         "Explanation: `Explanation for the answer based on the knowledge`. Answer based on explanation: `yes or no`.".
#
#         Comprehend following context: {context_passage}
#
#         Question: {q}
#         Ground-Truth Answers: {answers}
#         Candidate: {candidate_answer}
#
#         Is the candidate correct?
#     """
    # response = infer_vicuna(prompt)
    # if response.lower().rstrip(".").endswith("yes") or response.lower().startswith("yes"):
    #     acceptable = "Yes"
    # elif response.lower().rstrip(".").endswith("no") or response.lower().startswith("no"):
    #     acceptable = "No"
    # else:
    #     acceptable = ""
    #     logger.warning(f"Invalid response to `{q}` & `{candidate_answer}`: {response}")
    #     logger.warning(f"Prompt: {prompt}")
    #
    # return int(acceptable == "Yes"), response


def em_eval(
    gold_answers: Iterable[str], candidate_answer: str, match: str = "string", unacceptable_answers: Set[str] = None
) -> int:
    if not gold_answers:
        if unacceptable_answers and candidate_answer in unacceptable_answers:
            return 0
        else:
            return -1

    return int(
        metric_max_over_ground_truths(
            regex_match if match == "regex" else exact_match_score,
            candidate_answer,
            gold_answers,
        )
    )


def f1_eval(gold_answers: Iterable[str], candidate_answer: str, unacceptable_answers: Set[str] = None) -> float:
    if not gold_answers:
        if unacceptable_answers and candidate_answer in unacceptable_answers:
            return 0
        else:
            return -1

    return metric_max_over_ground_truths(
        f1_score,
        candidate_answer,
        gold_answers,
    )


def _prepare_data(
    predict_file: os.PathLike,
    dataset_file: Optional[os.PathLike] = None,
    annotation_file: Optional[os.PathLike] = None,
) -> List[Candidate]:
    if dataset_file is not None:
        questions = list(read_questions(dataset_file))
    else:
        questions = None

    if annotation_file and os.path.exists(annotation_file):
        annotated_answers = read_annotations(annotation_file)
    else:
        annotated_answers = {}

    predicted_dict, questions = read_predict_file(
        predict_file,
        questions,
    )

    candidates = []
    for question in tqdm(questions):
        qkey = question.tokenized_text.lower()
        if annotated_answers and qkey not in annotated_answers:
            continue

        if qkey not in predicted_dict:
            logger.warning(f"Question not found in prediction file and thus skipped: `{question.text}`")
            continue

        if not question.has_annotated_answers:
            logger.warning(f"Question with no annotated answers skipped: `{question.text}`")
            continue

        if annotated_answers:
            question.update_answers(annotated_answers[qkey])
        candidates.append(Candidate(predicted_dict[qkey], question))

    return candidates


def evaluate(
    question: str,
    candidate_answer: str,
    gold_answers: Union[Set[str], Sequence[str]],
    model_name: Optional[str] = None,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> Mapping[str, float]:
    q = Question(question, gold_answers)
    candidate = Candidate(candidate_answer, q)

    em = em_eval(gold_answers, candidate_answer)
    f1 = f1_eval(gold_answers, candidate_answer)

    result = dict(em=em, f1=f1)

    if model_name:
        if _is_openai_model(model_name):
            output = gpt_eval(model_name, [candidate], max_new_tokens=max_new_tokens, temperature=temperature)
        else:
            output = llm_eval(model_name, [candidate], max_new_tokens=max_new_tokens)

        acceptable, response = output[0]
        result[model_name] = acceptable
        result["response"] = response

    return result


def evaluate_file(
    predict_file: os.PathLike,
    dataset_file: Optional[os.PathLike] = None,
    annotation_file: Optional[os.PathLike] = None,
    output_file: Optional[os.PathLike] = None,
    prompt_file: Optional[str] = None,
    context_file: Optional[str] = None,
    model_name: Optional[str] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    batch_size: int = 1,
    do_greedy: bool = False,
    top_p: float = 1.0,
    num_gpus: Optional[int] = None,
    overwrite_cache: bool = False,
    return_per_sample: bool = False,
) -> Mapping[str, Union[float, List[float]]]:
    predict_file = Path(predict_file)
    if output_file:
        output_path = Path(output_file)
    else:
        output_name = f"{predict_file.stem}_eval"
        if model_name:
            output_name += f"-{model_name}"
        if annotation_file:
            annotation_name = Path(annotation_file).stem
            output_name += f"-{annotation_name[annotation_name.index('_') + 1:]}"

        output_path = predict_file.parent / f"{output_name}.tsv"

    candidates = _prepare_data(predict_file, dataset_file, annotation_file)
    if model_name:
        if _is_openai_model(model_name):
            eval_output = gpt_eval(
                model_name,
                candidates,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                experiment_name=output_path.stem,
                cache_dir=predict_file.parent,
                overwrite_cache=overwrite_cache,
            )
        else:
            eval_output = llm_eval(
                model_name,
                candidates,
                prompt_file=prompt_file,
                context_file=context_file,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                do_sample=not do_greedy,
                top_p=top_p,
                num_gpus=num_gpus,
            )
    else:
        eval_output = None

    eval_result = _calc_metrics(
        candidates, eval_output, model_name, annotation_file and os.path.exists(annotation_file)
    )
    if "AnnotatedEM" in eval_result and len(eval_result["EM"]) < len(candidates):
        logger.info(
            f"Only questions found in annotation file were evaluated: {len(eval_result['EM'])} out of {len(candidates)}"
        )

    _save_output(candidates, eval_output, eval_result, output_path)

    return {metric: scores if return_per_sample else np.mean(scores) for metric, scores in eval_result.items()}


def _calc_metrics(candidates: Sequence[Candidate], eval_output, model_name: str, has_annotated_file: bool = False):
    em_scores, f1_scores = [], []
    annotated_em_scores = []
    acceptables = []

    for i, candidate in enumerate(candidates):
        question = candidate.question

        em = em_eval(question.answers, candidate.answer)
        f1 = f1_eval(question.answers, candidate.answer)

        if em < 0 or f1 < 0:
            logger.warning(
                f"Predicted answer could not be evaluated: "
                f"`{question.text}` -> `{candidate.answer}` vs. {question.gold_answers}"
            )
            # continue

        if has_annotated_file:
            annotated_em = em_eval(question.gold_answers, candidate.answer, question.unacceptable_answers)
            if annotated_em < 0:
                logger.warning(
                    f"Predicted answer could not be evaluated after applying annotations: "
                    f"`{question.text}` -> `{candidate.answer}` vs. {question.gold_answers}"
                )
                # continue

            annotated_em_scores.append(annotated_em)

        em_scores.append(em)
        f1_scores.append(f1)

        if eval_output:
            acceptable = eval_output[i]
            acceptables.append(acceptable)

    eval_result = {
        "EM": em_scores,
        "F1": f1_scores,
        model_name: acceptables,
    }

    if annotated_em_scores:
        eval_result["AnnotatedEM"] = annotated_em_scores

    return eval_result


def _save_output(
    candidates: Sequence[Candidate],
    eval_result,
    model_output,
    output_file: os.PathLike,
):
    with open(output_file, "w") as f:
        w = csv.writer(f, delimiter="\t")
        headers = ["id", "Question", "Gold answers", "Model answer", "EM", "F1"]
        if "AnnotatedEM" in eval_result:
            headers.append("AnnotatedEM")

        for m in sorted(eval_result.keys()):
            if m not in ("EM", "F1", "AnnotatedEM"):
                headers.append(m)

        if model_output:
            headers.append("Response")

        w.writerow(headers)

        for i, (candidate, result) in tqdm(enumerate(zip(candidates, eval_result))):
            question = candidate.question
            predicted_answer = candidate.answer

            row = [question.id, question.text, question.answers, predicted_answer, result["EM"], result["F1"]]

            if "AnnotatedEM" in eval_result:
                row.append(result["AnnotatedEM"])

            for m in sorted(eval_result.keys()):
                if m not in ("EM", "F1", "AnnotatedEM"):
                    row.append(eval_result[m])

            if model_output:
                _, out = model_output[i]
                row.append(out)

            w.writerow(row)
