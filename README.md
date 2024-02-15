# Evaluating Question Answering with LLMs

<!-- Thanks for your interest in our repo! -->
<!-- We were inspired by FaithDial to organize this repo! 🖖 -->

## Quick Links

  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Data](#data)
  - [Evaluation](#evaluation)
  - [Bugs or Questions?](#bugsbug-or-questionsquestion)
  - [Citation](#citation)

## Overview
Lexical matching is the standard evaluation method for open-domain question answering (QA), 
but it fails when plausible answers are not in the provided list.
In this repo, we use open-source and proprietary LLMs for evaluation.

### Requirements
The code needs Python 3.8+ (we tested it with Python `3.8`).

To install from the repo:
```bash
pip install git+https://github.com/ehsk/QA-eval.git
```

To install from the source:
```bash
git clone git@github.com:ehsk/QA-eval.git
pip install -e .
```

## Data
We worked on the [Natural Questions-open](https://huggingface.co/datasets/nq_open) [(Lee et al., ACL 2019)](https://aclanthology.org/P19-1612) test dataset that consists of 3,610 questions. We randomly sampled 301 questions for annotation.

Taken from [here](https://github.com/ehsk/OpenQA-eval), we provide the answers generated by QA models along with the output of the four evaluation mechanisms in the [data](data) directory:

```bash
   data
    ├── model_outputs                                   # Answers generated by 12 open-domain QA models
    │   ├── NQ301_text-davinci-003_fewshot-n64.jsonl    # InstructGPT (few-shot)
    │   ├── NQ301_text-davinci-003_zeroshot.jsonl       # InstructGPT (zero-shot)
    │   ├── NQ_ANCE-plus_FiD.jsonl                      # ANCE+ & Fusion-In-Decoder
    │   └── ...
    ├── NQ301_BEM.tsv                                   # BEM predictions for all generated answers
    ├── NQ301_gpt-4.tsv                                 # GPT4-eval output for all generated answers
    ├── NQ301_human.tsv                                 # Human judgments for all generated answers
    └── NQ301_text-davinci-003.tsv                      # InstructGPT-eval output for all generated answers
```

The annotations can also be viewed online [here](https://docs.google.com/spreadsheets/d/1X0SpOg4Y1BCuNnGxwr-fqjA9tn1Y8XRkheUa_49QTgY/edit?usp=sharing).

## Evaluation

The evaluation script takes a prediction file in a jsonl format as below and measures its performance with different metrics.

```json lines
{"question": "who is under the mask of darth vader", "answer": ["Anakin Skywalker"], "prediction": "Anakin Skywalker"}
{"question": "which is the default file extension for an audio file in windows media player", "answer": ["Windows Playlist ( WPL )"], "prediction": "WMA"}
```

The following command computes only two lexical matching metrics: EM (Exact-Match accuracy) and macro-averaged F1.
```bash
python -m qaeval /path/to/prediction_file.jsonl
```

To evaluate using an LLM like InstructGPT-eval in the paper, the model name (`text-davinci-003` or `gpt-4`) argument should be passed:
```bash
python -m qaeval /path/to/prediction_file.jsonl --model text-davinci-003
```
which calls OpenAI APIs. The environment variable `OPENAI_API_KEY` needs to be set first. 
*Bear in mind that running this command will result in charges to your OpenAI account.* 
We did not see a significant difference between GPT-4 and InstructGPT, so we recommend using the cheaper model (InstructGPT).

To evaluate using our provided annotated files including human judgment, you can simply run:
```bash
python -m qaeval /path/to/prediction_file.jsonl --annotation data/NQ301_human.tsv
```
The above command evaluates only 301 annotated questions and skips the rest in the prediction file.

## Bugs:bug: or questions:question:

If you have any questions or encounter any problems, feel free to open an issue.


## License

This work is licensed under the MIT license. See [LICENSE](LICENSE) for details.
