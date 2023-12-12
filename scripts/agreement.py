import argparse
import csv
from pathlib import Path
from  nltk.metrics import agreement
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
for i, results_file in tqdm(enumerate(results_dir.glob(file_pattern.name)), colour="green"):
    with open(results_file) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r, row in enumerate(reader):
            question = row[1]
            gold_answer = eval(row[2])
            prediction = row[3]
            judgments = eval(row[8])
            for j, judg in enumerate(judgments):
                annotations.append((f"c{j}", f"{results_file.stem}#{r}", judg))


print(f"{len(annotations)} collected from {len(list(results_dir.glob(file_pattern.name)))} result files")

task = agreement.AnnotationTask(data=annotations)
print("Fleiss' Kappa =", task.multi_kappa())

