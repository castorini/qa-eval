import argparse
import csv
from collections import defaultdict, Counter
from pathlib import Path
from nltk.metrics import agreement
from tqdm import tqdm

from qaeval import SimpleTokenizer

FAILURE_MODES = {
    "Semantic Equivalence": (
        "Multinominal Entities",
        "EM in Explanatory Answer",
        "Bridging/Abridging",
        "More Elaborate Answer",
        "Tokenization Mismatches",
        "Synonymous Answer",
    ),
    "Symbolic Equivalence": ("Symbolic Eq", "Failure in Symbolic Eq"),
    "Granularity Discrepancy": ("Temporal Granularity Discrepancy", "Spatial Granularity Discrepancy"),
    "Incomplete Reference Answers": ("List", "Open-ended", "Compound"),
    "Intrinsic Ambiguity": ("Intrinsic Ambiguity",),
    "Incorrect Gold Answers": ("Incorrect Gold Answers",),
}
BOTTOMUP_FAILURE_MODES = {child: parent for parent, children in FAILURE_MODES.items() for child in children}

parser = argparse.ArgumentParser()
parser.add_argument(
    "file_pattern",
    type=str,
    help="File pattern",
)
parser.add_argument(
    "--analysis_file",
    type=str,
    required=True,
    help="Path to analysis file",
)

args = parser.parse_args()

tokenizer = SimpleTokenizer()

analytics = {}
human_annotation = {}
with open(args.analysis_file) as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        q = tokenizer.tokenize(row["Question"], as_string=True).lower()
        p = tokenizer.tokenize(row["Predicted answer"], as_string=True)
        acceptable = int(row["Acceptable-Hu"].strip().lower() == "yes")
        reason = row["Why-Hu"].strip()
        if reason:
            analytics[f"{q}###{p}"] = reason

        human_annotation[f"{q}###{p}"] = acceptable

print(f"{len(analytics)} records / {len(human_annotation)} annotations loaded from '{args.analysis_file}'")

file_pattern = Path(args.file_pattern)
results_dir = file_pattern.parent

predictions = {}
for i, results_file in tqdm(enumerate(results_dir.glob(file_pattern.name)), desc="loading", colour="green"):
    with open(results_file) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r, row in enumerate(reader):
            question = row[1]
            prediction = row[3]
            exact_match = int(row[4])
            final_judgment = int(row[6])
            judgments = eval(row[8])

            tok_q = tokenizer.tokenize(question, as_string=True).lower()
            tok_p = tokenizer.tokenize(prediction, as_string=True)
            key = f"{tok_q}###{tok_p}"
            if key not in predictions:
                predictions[key] = {
                    "question": question,
                    "prediction": prediction,
                    "EM": exact_match,
                    "judgment": final_judgment,
                    "judgments": judgments,
                }
            else:
                assert (
                    predictions[key]["judgment"] != final_judgment
                ), f"inconsistency found in '{results_file}': [{question}] [{prediction}]"

print(
    f"{len(predictions)} QA predictions collected from {len(list(results_dir.glob(file_pattern.name)))} result files"
)

em_misses = 0
fp = 0
fn = 0
agreements_per_em_failure = defaultdict(int)
judg1_counts = defaultdict(int)
disagreements_per_diverging_em_failure = defaultdict(lambda: defaultdict(int))
for qa_key, pred in tqdm(predictions.items(), desc="collecting stats", colour="yellow"):
    exact_match = pred["EM"]
    final_judgment = pred["judgment"]
    judgments = eval(row[8])

    n_judg1 = Counter(pred["judgments"]).get(1, 0)
    n_judg1 = round(n_judg1 / len(judgments), 1)
    judg1_counts[n_judg1] += 1

    reason = analytics.get(key, None)
    if reason:
        failure_mode = BOTTOMUP_FAILURE_MODES.get(reason, reason)

    if final_judgment != human_annotation[key]:
        if final_judgment:
            fp += 1
        else:
            if exact_match:
                em_misses += 1
            fn += 1

        if reason:
            disagreements_per_diverging_em_failure[failure_mode][n_judg1] += 1
    else:
        if reason:
            agreements_per_em_failure[failure_mode] += 1

total_diverging_freq = sum(freq for n_judg1, freq in judg1_counts.items() if n_judg1 != 0 and n_judg1 != 1)

print()
print("***" * 30)
print()

print(f"FP: {fp} ({100. * fp / len(predictions):.2f}%)")
print(f"FN: {fn} ({100. * fn / len(predictions):.2f}%)")
print(f"  Exact-Match misses: {em_misses} ({100. * em_misses / fn:.2f}%)")

print()
print("***" * 30)
print()

for failure_mode in sorted(agreements_per_em_failure.keys()):
    freq = agreements_per_em_failure[failure_mode]
    print(
        f"{failure_mode}: {freq} "
        f"(Out of EM failures: {100. * freq / len(analytics):.2f}%) "
        f"(Out of all: {100. * freq / len(predictions):.2f}%)"
    )

print()
print("***" * 30)
print()

for failure_mode in sorted(disagreements_per_diverging_em_failure.keys()):
    total_fail = sum(disagreements_per_diverging_em_failure[failure_mode].values())
    diverging_freq = sum(
        freq
        for n_judg1, freq in disagreements_per_diverging_em_failure[failure_mode].items()
        if n_judg1 != 0 and n_judg1 != 1
    )

    print(failure_mode)
    for n_judg1 in sorted(disagreements_per_diverging_em_failure[failure_mode].keys()):
        freq = disagreements_per_diverging_em_failure[failure_mode][n_judg1]
        print(f"\t[{n_judg1}] {freq} ({100. * freq / total_fail:.2f}%) (Out of FN: {100. * freq / fn:.2f}%)")
    print("---")
    print(
        f"\t~~~ {diverging_freq} ({100. * diverging_freq / total_fail:.2f}%) "
        f"(Out of FN: {100. * diverging_freq / fn:.2f}%) "
        f"(Out of diverges: {100. * diverging_freq / total_diverging_freq:.2f}%)"
    )

print()
print("***" * 30)
print()

for n_judg1 in sorted(judg1_counts.keys()):
    print(
        f"#judged {n_judg1} times as one = {judg1_counts[n_judg1]} "
        f"({100. * judg1_counts[n_judg1] / sum(judg1_counts.values()):.1f}%)"
    )
print("---")
print(f"#judged diverged = {total_diverging_freq} " f"({100. * total_diverging_freq / sum(judg1_counts.values()):.1f}%)")
