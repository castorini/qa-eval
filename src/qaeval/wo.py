import logging

from sacrebleu.metrics import BLEU
import evaluate

logger = logging.getLogger("wo")


def wo_eval(candidates, **kwargs):
    max_refs = max([len(c.question.answers) for c in candidates])

    rouge = evaluate.load("rouge")
    hf_bleu = evaluate.load("bleu")

    cands = []
    padded_refs = []
    refs = []
    for candidate in candidates:
        cands.append(candidate.answer)

        padded_golds = [""] * max_refs
        golds = []
        for i, gold_answer in enumerate(candidate.question.answers):
            padded_golds[i] = gold_answer
            golds.append(gold_answer)
        padded_refs.append(padded_golds)
        refs.append(golds)

    bleu = BLEU()
    score = bleu.corpus_score(cands, refs).score
    rouge_scores = rouge.compute(predictions=cands, references=refs)
    hf_score = hf_bleu.compute(predictions=cands, references=refs)["bleu"]

    return {
        "sacreBLEU": score,
        "BLEU": hf_score,
        "RougeL": rouge_scores["rougeL"],
        "Rouge1": rouge_scores["rouge1"],
        "Rouge2": rouge_scores["rouge2"],
    }
