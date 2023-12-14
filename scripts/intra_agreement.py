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
    default=None,
    help="Path to analysis file",
)

args = parser.parse_args()

tokenizer = SimpleTokenizer()

analytics = {}
if args.analysis_file:
    with open(args.analysis_file) as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            q = tokenizer.tokenize(row["Question"], as_string=True).lower()
            p = tokenizer.tokenize(row["Predicted answer"], as_string=True)
            reason = row["Why-Hu"].strip()
            if reason:
                analytics[f"{q}###{p}"] = reason

    print(f"{len(analytics)} records loaded from '{args.analysis_file}'")

file_pattern = Path(args.file_pattern)
results_dir = file_pattern.parent

annotations = []
judg1_counts = defaultdict(int)
judg1_ratios = defaultdict(int)
judg1_per_reason = defaultdict(lambda: defaultdict(int))
for i, results_file in tqdm(enumerate(results_dir.glob(file_pattern.name)), colour="green"):
    with open(results_file) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r, row in enumerate(reader):
            question = row[1]
            gold_answer = eval(row[2])
            prediction = row[3]
            judgments = eval(row[8])
            judged_ones = Counter(judgments).get(1, 0)
            judg1_counts[judged_ones] += 1
            judg1_ratios[round(judged_ones / len(judgments), 1)] += 1
            for j, judg in enumerate(judgments):
                annotations.append((f"c{j}", f"{results_file.stem}#{r}", judg))

            tok_q = tokenizer.tokenize(question, as_string=True).lower()
            tok_p = tokenizer.tokenize(prediction, as_string=True)
            reason = analytics.get(f"{tok_q}###{tok_p}", None)
            if reason:
                failure_mode = BOTTOMUP_FAILURE_MODES.get(reason, reason)
                judg1_per_reason[failure_mode][judged_ones] += 1

print(f"{len(annotations)} collected from {len(list(results_dir.glob(file_pattern.name)))} result files")

task = agreement.AnnotationTask(data=annotations)
print("Fleiss' Kappa =", task.multi_kappa())

print()
print("***" * 30)
print()

for freq in sorted(judg1_counts.keys()):
    print(
        f"#judged {freq} times as one = {judg1_counts[freq]} "
        f"({100. * judg1_counts[freq] / sum(judg1_counts.values()):.1f}%)"
    )

print()
print("***" * 30)
print()

for ratio in sorted(judg1_ratios.keys()):
    print(
        f"#judged {ratio} times as one = {judg1_ratios[ratio]} "
        f"({100. * judg1_ratios[ratio] / sum(judg1_ratios.values()):.1f}%)"
    )

print()
print("***" * 30)
print()

for failure_mode, counts in judg1_per_reason.items():
    _percent = 100.0 * sum(counts.values()) / sum(judg1_counts.values())
    for freq in sorted(counts.keys()):
        print(
            f"[{failure_mode} ({_percent:.1f}%)] "
            f"#judged {freq} times as one = {counts[freq]} "
            f"({100. * counts[freq] / sum(counts.values()):.1f}%)"
        )

print()
print("***" * 30)
