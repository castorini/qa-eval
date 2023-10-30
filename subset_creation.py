import jsonlines

# Define the file paths
file1_path = "data/model_outputs/NQ301_text-davinci-003_zeroshot.jsonl"
file2_path = "data/model_outputs/NQ_Rocketv2_FiD.jsonl"
output_path = "data/model_outputs/NQ301_Rocketv2_FiD.jsonl"  # Define the output file path

# Create a set to store questions from file1.jsonl
questions_set = set()

# Read questions from file1.jsonl and store them in the set
with jsonlines.open(file1_path) as reader:
    for item in reader:
        if "question" in item:
            questions_set.add(item["question"])

# Create a list to store JSON objects from file2.jsonl where the question is in the set
subset = []

# Read file2.jsonl and extract questions that are in the set
with jsonlines.open(file2_path) as reader:
    for item in reader:
        if "question" in item and item["question"] in questions_set:
            subset.append(item)

# Save the extracted subset to a new JSONL file
with jsonlines.open(output_path, mode='w') as writer:
    for item in subset:
        writer.write(item)
