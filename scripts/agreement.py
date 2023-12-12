import argparse
import csv
from collections import defaultdict, Counter
from pathlib import Path
from nltk.metrics import agreement
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "file_pattern",
    type=str,
    help="File pattern",
)

args = parser.parse_args()

file_pattern = Path(args.file_pattern)
results_dir = file_pattern.parent

annotations = []
judg1_counts = defaultdict(int)
judg1_ratios = defaultdict(int)
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


print(f"{len(annotations)} collected from {len(list(results_dir.glob(file_pattern.name)))} result files")

task = agreement.AnnotationTask(data=annotations)
print("Fleiss' Kappa =", task.multi_kappa())

print("***" * 30)
for freq in sorted(judg1_counts.keys()):
    print(
        f"#judged {freq} times as one = {judg1_counts[freq]} "
        f"({100. * judg1_counts[freq] / sum(judg1_counts.values()):.1f}%)"
    )

print()
print("***" * 30)
for ratio in sorted(judg1_ratios.keys()):
    print(
        f"#judged {ratio} times as one = {judg1_ratios[ratio]} "
        f"({100. * judg1_ratios[ratio] / sum(judg1_ratios.values()):.1f}%)"
    )
