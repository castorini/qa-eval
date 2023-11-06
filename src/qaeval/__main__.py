"""
Basic usage:
python -m qaeval /path/to/predict_file.jsonl
"""
import argparse
import logging
import numpy as np

from .eval import evaluate_file

logging.basicConfig(
    level=logging.INFO, format="%(levelname).1s %(asctime)s [ %(message)s ]", handlers=[logging.StreamHandler()]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predict_file",
        type=str,
        help="Path to predict file",
    )
    parser.add_argument("--dataset", type=str, default=None, help="dataset path")
    parser.add_argument(
        "--annotation", type=str, default=None, help="tsv file including additional answer annotations"
    )
    parser.add_argument(
        "--contexts",
        type=str,
        default=None,
        help="Context file containing retrieved passages (used only for public models)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompts/eval-v1.0.txt",
        help="Prompt template file (used only for public models)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=(
            "text-davinci-003",
            "gpt-3.5-turbo",
            "gpt-4",
            "lmsys/vicuna-13b-v1.5-16k",
            "lmsys/vicuna-7b-v1.5-16k",
            "lmsys/vicuna-7b-v1.5",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "HuggingFaceH4/zephyr-7b-beta",
        ),
        help="Model names",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (used in OpenAI API)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature used in generation (used in OpenAI API)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for running inference on public models",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs for inference, default is all available GPUs (used only for public models)",
    )
    parser.add_argument(
        "--do_greedy",
        action="store_true",
        default=False,
        help="Whether to disable sampling in decoding (used only for public models)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Probability mass for nucleus sampling (used only for public models)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output file containing API outputs (default: will be saved in `--predict_file` directory in a tsv format)",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        default=False,
        help="Whether to overwrite cached evaluation results from a previous run (only when OpenAI API used)",
    )

    args = parser.parse_args()

    result = evaluate_file(
        args.predict_file,
        args.dataset,
        args.annotation,
        args.output,
        args.prompt,
        args.contexts,
        args.model,
        args.max_tokens,
        args.temperature,
        args.batch_size,
        args.do_greedy,
        args.top_p,
        args.num_gpus,
        args.overwrite_cache,
        return_per_sample=True,
    )

    em_scores = result["EM"]
    f1_scores = result["F1"]

    print(f"EM: {100.0 * np.mean(em_scores):.2f} ({np.sum(em_scores)}/{len(em_scores)})")
    print(f"F1: {100.0 * np.mean(f1_scores):.2f}")

    if "AnnotatedEM" in result:
        annotated_em_scores = result["AnnotatedEM"]
        print(
            f"AnnotatedEM: {100.0 * np.mean(annotated_em_scores):.2f} ({np.sum(annotated_em_scores)}/{len(annotated_em_scores)})"
        )

    if args.model in result:
        gpt_scores = result[args.model]
        print(f"{args.model}: {100.0 * np.mean(gpt_scores):.2f} ({np.sum(gpt_scores)}/{len(gpt_scores)})")


if __name__ == "__main__":
    main()
