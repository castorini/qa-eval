"""
*** This script calls OpenAI APIs that will charge you per use

OPENAI_API_KEY env variable should be set to run this script
"""
import csv
import json
import logging
import os
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

from .data_utils import read_questions, read_predict_file, read_annotations, Question, SimpleTokenizer
from .llm import OpenAIProxy
from .vicuna_llm import infer_vicuna
from .squad_evaluate import metric_max_over_ground_truths, regex_match, exact_match_score, f1_score

logger = logging.getLogger("eval")


def gpt_eval(question: Question, candidate_answer: str, openai_proxy: OpenAIProxy) -> Tuple[int, str]:
    answers = " or ".join(question.answers)
    q = question.text

    if not q.endswith("?"):
        q += "?"

    prompt = f"Question: {q}\nAnswer: {answers}\nCandidate: {candidate_answer}\n\nIs candidate correct?"
    response = openai_proxy(prompt)
    if response.lower().startswith("yes"):
        acceptable = "Yes"
    elif response.lower().startswith("no"):
        acceptable = "No"
    else:
        acceptable = ""
        logger.warning(f"Invalid response to `{q}` & `{candidate_answer}`: {response}")
        logger.warning(f"Prompt: {prompt}")

    return int(acceptable == "Yes"), response

def vicuna_eval(question: Question, candidate_answer: str) -> Tuple[int, str]:
    answers = " or ".join(question.answers)
    q = question.text

    with open('../passage_dpr.json', 'r') as json_file:
        context_passage = json.load(json_file)

    passage = context_passage.get(q, {}).get('contents')

    if not q.endswith("?"):
        q += "?"

    prompt = f"""
        You are an expert judge of a content. You'll be give a question, ground-truth answers
        and a candidate answer. Using your inbuilt knowledge, given context and given ground-truth answers try to 
        verify if the candidate is correct or not. Provide explanation for the comparison and provide
        answer based on the explanation as "yes" or "no". Here, "yes" represents that the candidate answer
        is relevant and correct based on either inbuilt knowledge, given context or given ground-truth answers. If not,
        the answer based on the explanation would be "no".
        
        Here are some of the sample examples:

        Question: how long have we been celebrating pi day
        Ground-Truth Answers: "1988", "2009"
        Candidate: We have been celebrating pi day since 1988.
        
        Is the candidate correct?
        Since 1998 we have been celebrating pi day. Answer based on explanation: yes.
        
        ###
        Question: who plays dylan in daddy's home 2
        Ground-Truth Answers: "Owen Vaccaro"
        Candidate: Vaccaro
        
        Is the candidate correct?
        Owen Vaccaro plays Dylan in the movie "Daddy's Home 2". Vaccaro is the last name. Answer based on explanation: yes.
                
        ###
        Now make prediction for following data.
        Compulsory format for the `Is the candidate correct?` question's answer: 
        "Explanation: `Explanation for the answer based on the knowledge`. Answer based on explanation: `yes or no`.".

        Comprehend following context: {context_passage}

        Question: {q}
        Ground-Truth Answers: {answers}
        Candidate: {candidate_answer}
        
        Is the candidate correct?
    """
    response = infer_vicuna(prompt)
    if response.lower().rstrip(".").endswith("yes") or response.lower().startswith("yes"):
        acceptable = "Yes"
    elif response.lower().rstrip(".").endswith("no") or response.lower().startswith("no"):
        acceptable = "No"
    else:
        acceptable = ""
        logger.warning(f"Invalid response to `{q}` & `{candidate_answer}`: {response}")
        logger.warning(f"Prompt: {prompt}")

    return int(acceptable == "Yes"), response


def em_eval(question: Question, candidate_answer: str, match: str = "string") -> int:
    if not question.gold_answers:
        if question.is_unacceptable(candidate_answer):
            return 0
        else:
            return -1

    return int(
        metric_max_over_ground_truths(
            regex_match if match == "regex" else exact_match_score,
            candidate_answer,
            question.gold_answers,
        )
    )


def f1_eval(question: Question, candidate_answer: str) -> float:
    if not question.gold_answers:
        if question.is_unacceptable(candidate_answer):
            return 0
        else:
            return -1

    return metric_max_over_ground_truths(
        f1_score,
        candidate_answer,
        question.gold_answers,
    )


def _load_evaluated(output_file: os.PathLike) -> Mapping[str, Tuple[int, str]]:
    tokenizer = SimpleTokenizer()

    cached = {}
    with open(output_file, "r") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)

        for row in r:
            if len(row) < 8:
                continue

            gpt = int(row[6])
            resp = row[7]

            q = tokenizer.tokenize(row[1], as_string=True).lower()
            ans = row[2]
            cached[f"{q}|{ans}"] = (gpt, resp)

    return cached


def evaluate(
    question: str,
    candidate_answer: str,
    gold_answers: Union[Set[str], Sequence[str]],
    openai_proxy: Optional[OpenAIProxy] = None,
    openai_model: str = "text-davinci-003",
    max_tokens: int = 100,
    temperature: float = 1.0,
) -> Mapping[str, float]:

    if openai_proxy is None:
        openai_proxy = OpenAIProxy(openai_model, max_tokens, temperature)

    q = Question(question, gold_answers)
    gpt_result, gpt_response = gpt_eval(q, candidate_answer, openai_proxy)
    em = em_eval(q, candidate_answer)
    f1 = f1_eval(q, candidate_answer)

    return {"em": em, "f1": f1, openai_proxy.model_name: gpt_result}


def evaluate_file(
    predict_file: os.PathLike,
    dataset_file: Optional[os.PathLike] = None,
    annotation_file: Optional[os.PathLike] = None,
    output_file: Optional[os.PathLike] = None,
    openai_model: Optional[str] = None,
    openai_model_bool: Optional[bool] = False,
    max_tokens: int = 100,
    temperature: float = 0.0,
    overwrite_cache: bool = False,
    return_per_sample: bool = False,
) -> Mapping[str, Union[float, List[float]]]:
    predict_file = Path(predict_file)

    if output_file:
        output_path = Path(output_file)
    else:
        output_name = f"{predict_file.stem}_eval"
        if openai_model:
            output_name += f"-{openai_model}"
        if annotation_file:
            annotation_name = Path(annotation_file).stem
            output_name += f"-{annotation_name[annotation_name.index('_') + 1:]}"

        output_path = predict_file.parent / f"{output_name}.tsv"

    if openai_model and openai_model_bool:
        openai_proxy = OpenAIProxy(openai_model, max_tokens, temperature)
    elif openai_model:
        openai_proxy = openai_model
    else:
        openai_proxy = None

    if dataset_file is not None:
        questions = list(read_questions(dataset_file))
    else:
        questions = None

    eval_result = _evaluate(predict_file, questions, openai_proxy, output_path, annotation_file, overwrite_cache)
    questions = eval_result.pop("questions")

    if openai_proxy and openai_model_bool:
        logger.info(f"OpenAI API call stats: {openai_proxy.get_stats()}")

    if "AnnotatedEM" in eval_result and len(eval_result["EM"]) < len(questions):
        logger.info(
            f"Only questions found in annotation file were evaluated: {len(eval_result['EM'])} out of {len(questions)}"
        )

    return {metric: scores if return_per_sample else np.mean(scores) for metric, scores in eval_result.items()}


def _evaluate(
    predict_file: os.PathLike,
    questions: Optional[Sequence[Question]],
    openai_proxy: Optional[OpenAIProxy],
    output_file: os.PathLike,
    annotation_file: Optional[os.PathLike] = None,
    overwrite_cache: bool = False,
) -> Mapping[str, list]:
    predicted_dict, questions = read_predict_file(
        predict_file,
        questions,
    )

    if annotation_file and os.path.exists(annotation_file):
        annotated_answers = read_annotations(annotation_file)
    else:
        annotated_answers = {}

    cached_output = {}
    if os.path.exists(output_file) and not overwrite_cache:
        cached_output = _load_evaluated(output_file)

    em_scores, f1_scores = [], []
    annotated_em_scores = []
    gpt_scores = []

    with open(output_file, "w") as f:
        w = csv.writer(f, delimiter="\t")
        headers = ["id", "Question", "Gold answers", "Model answer", "EM", "F1"]
        if annotated_answers:
            headers.append("AnnotatedEM")

        if isinstance(openai_proxy, OpenAIProxy):
            headers.extend([openai_proxy.model_name, "Response"])
        elif isinstance(openai_proxy, str):
            headers.extend([openai_proxy, "Response"])

        w.writerow(headers)

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

            predicted_answer = predicted_dict[qkey]
            em = em_eval(question, predicted_answer)
            f1 = f1_eval(question, predicted_answer)

            if em < 0 or f1 < 0:
                logger.warning(
                    f"Predicted answer could not be evaluated: `{question.text}` -> `{predicted_answer}` vs. {question.gold_answers}"
                )
                continue

            row = [question.id, question.text, question.answers, predicted_answer, em, f1]

            if annotated_answers and qkey in annotated_answers:
                question.update_answers(annotated_answers[qkey])
                annotated_em = em_eval(question, predicted_answer)
                if annotated_em < 0:
                    logger.warning(
                        f"Predicted answer could not be evaluated after applying annotations: `{question.text}` -> `{predicted_answer}` vs. {question.gold_answers}"
                    )
                    continue

                annotated_em_scores.append(annotated_em)
                row.append(annotated_em)

            if openai_proxy:
                cache_key = f"{question.text}|{predicted_answer}"
                if cache_key not in cached_output:
                    if isinstance(openai_proxy, OpenAIProxy):
                        gpt_result, gpt_response = gpt_eval(question, predicted_answer, openai_proxy)
                    else:
                        gpt_result, gpt_response = vicuna_eval(question, predicted_answer)
                else:
                    gpt_result, gpt_response = cached_output[cache_key]

                gpt_scores.append(gpt_result)
                row.extend((gpt_result, gpt_response))

            em_scores.append(em)
            f1_scores.append(f1)

            w.writerow(row)
            f.flush()

    eval_result = {
        "questions": questions,
        "EM": em_scores,
        "F1": f1_scores,
    }

    if annotated_answers:
        eval_result["AnnotatedEM"] = annotated_em_scores

    if openai_proxy:
        if isinstance(openai_proxy, OpenAIProxy):
            eval_result[openai_proxy.model_name] = gpt_scores
        else:
            eval_result[openai_proxy] = gpt_scores

    return eval_result
