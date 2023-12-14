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
    "Symbolic Equivalence": ("Symbolic Eq",),
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
        p = tokenizer.tokenize(row["Predicted answer"], as_string=True).lower()
        acceptable = int(row["Acceptable-Hu"].strip().lower() == "yes")
        reason = row["Why-Hu"].strip()
        if reason:
            analytics[f"{q}###{p}"] = BOTTOMUP_FAILURE_MODES.get(reason, reason)

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
            responses = row[7]
            if len(row) > 8:
                judgments = eval(row[8])
            else:
                judgments = [final_judgment]

            tok_q = tokenizer.tokenize(question, as_string=True).lower()
            tok_p = tokenizer.tokenize(prediction, as_string=True).lower()
            key = f"{tok_q}###{tok_p}"
            if key not in predictions:
                predictions[key] = {
                    "question": question,
                    "prediction": prediction,
                    "EM": exact_match,
                    "judgment": final_judgment,
                    "judgments": judgments,
                    "source": results_file.name,
                    "responses": responses,
                }
            else:
                if predictions[key]["judgment"] != final_judgment:
                    print(f"[WARN] inconsistency found for [{question}] [{prediction}]")
                    print(f"   [{results_file.name}]: {responses}")
                    print(f"   [{predictions[key]['source']}]: {predictions[key]['responses']}")
                    print("***" * 4)

print(
    f"{len(predictions)} QA predictions collected from {len(list(results_dir.glob(file_pattern.name)))} result files"
)

em_misses = 0
fp = 0
fn = 0
judg1_counts = defaultdict(int)
agreements_for_em_failures = defaultdict(lambda: defaultdict(int))
disagreements_for_em_failures = defaultdict(int)
exactmatch_fn = 0
for qa_key, pred in tqdm(predictions.items(), desc="collecting stats", colour="yellow"):
    exact_match = pred["EM"]
    final_judgment = pred["judgment"]
    judgments = pred["judgments"]

    n_judg1 = Counter(judgments).get(1, 0)
    n_judg1 = round(n_judg1 / len(judgments), 1)
    judg1_counts[n_judg1] += 1

    failure_mode = analytics.get(qa_key, None)

    if qa_key not in human_annotation:
        human_annotation[qa_key] = 0

    if human_annotation[qa_key] and not exact_match:
        exactmatch_fn += 1

    if final_judgment != human_annotation[qa_key]:
        if final_judgment:
            fp += 1
        else:
            if exact_match:
                em_misses += 1
            fn += 1

        if failure_mode:
            disagreements_for_em_failures[failure_mode] += 1
    else:
        if failure_mode:
            agreements_for_em_failures[failure_mode][n_judg1] += 1

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

print("Humans judge yes, but automated eval says yes")
print()
for failure_mode in sorted(agreements_for_em_failures.keys()):
    diverging_freq = sum(
        freq for n_judg1, freq in agreements_for_em_failures[failure_mode].items() if n_judg1 != 0 and n_judg1 != 1
    )

    total_freq = sum(agreements_for_em_failures[failure_mode].values())

    print(
        failure_mode,
        "---",
        f"{total_freq / len(analytics):.1f}%",
        f"({total_freq} out of {len(analytics)})",
        "---",
        "#Diverging",
        f"{diverging_freq}",
        f"({100. * diverging_freq / total_freq:.1f}%)",
    )

    for n_judg1 in sorted(agreements_for_em_failures[failure_mode].keys()):
        freq = agreements_for_em_failures[failure_mode][n_judg1]
        print(
            f"    [{n_judg1}] {100. * freq / total_freq:.1f}%",
            f"({freq} out of {total_freq})",
            "---",
            f"{100. * freq / len(analytics):.1f}%",
        )

print()
print("***" * 30)
print()

print("Humans judge yes, but automated eval says no")
print()
for failure_mode, freq in sorted(disagreements_for_em_failures.items(), key=lambda x: x[1], reverse=True):
    print(failure_mode, "---", f"({100. * freq / len(analytics):.1f}%)", f"({freq} out of {len(analytics)})")

print()
print("***" * 30)
print()

for n_judg1 in sorted(judg1_counts.keys()):
    print(
        f"#judged {n_judg1} times as one = {judg1_counts[n_judg1]} "
        f"({100. * judg1_counts[n_judg1] / sum(judg1_counts.values()):.1f}%)"
    )

if total_diverging_freq > 0:
    print("---")
    print(
        f"#judged diverged = {total_diverging_freq} "
        f"({100. * total_diverging_freq / sum(judg1_counts.values()):.1f}%)"
    )
