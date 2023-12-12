import logging

from sacrebleu.metrics import BLEU

logger = logging.getLogger("wo")


def wo_eval(candidates, **kwargs):
    max_refs = max([len(c.question.answers) for c in candidates])
    cands = []
    refs = []
    for candidate in candidates:
        cands.append(candidate.answer)

        golds = [""] * max_refs
        for i, gold_answer in enumerate(candidate.question.answers):
            golds[i] = gold_answer
        refs.append(golds)

    bleu = BLEU()
    score = bleu.corpus_score(cands, refs).score
    return score
