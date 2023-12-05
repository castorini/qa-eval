import argparse
import json
from pathlib import Path

# Define the file paths
parser = argparse.ArgumentParser()
parser.add_argument(
    "predict_file",
    type=str,
    help="Path to a predict file",
)
parser.add_argument(
    "--ref_file",
    type=str,
    default="data/model_outputs/NQ301_text-davinci-003_zeroshot.jsonl",
    help="Path to a reference predict file",
)

args = parser.parse_args()

# file1_path = "../data/model_outputs/NQ301_text-davinci-003_zeroshot.jsonl"
# file2_path = "../data/model_outputs/NQ_Rocketv2_FiD.jsonl"
# output_path = "../data/model_outputs/NQ301_Rocketv2_FiD.jsonl"  # Define the output file path

# Create a set to store questions from file1.jsonl
questions_set = set()

# Read questions from file1.jsonl and store them in the set
ref_file = Path(args.ref_file)
with open(ref_file) as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)
        if "question" in item:
            questions_set.add(item["question"])

# Create a list to store JSON objects from file2.jsonl where the question is in the set
subset = []

# Read file2.jsonl and extract questions that are in the set
predict_file = Path(args.predict_file)
with open(predict_file) as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)
        if "question" in item and item["question"] in questions_set:
            subset.append(item)

output_file = ref_file.stem.split("_")[0] + "_" + "_".join(predict_file.name.split("_")[1:])
output_path = predict_file.parent / output_file
# Save the extracted subset to a new JSONL file
with open(output_path, mode='w') as writer:
    for item in subset:
        writer.write(json.dumps(item) + "\n")

print(f"Output saved to '{output_path}'")
