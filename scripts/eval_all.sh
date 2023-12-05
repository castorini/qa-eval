#!/usr/bin/env bash

# scripts/eval_all.sh "0" 0 prompts/eval-v0.2-few-shot_chat.txt google/flan-t5-base 3
# scripts/eval_all.sh "1" 0 prompts/eval-v0.2-few-shot_chat.txt HuggingFaceH4/zephyr-7b-beta 3

Green='\033[0;32m'
NC='\033[0m' # No Color

CUDA_VISIBLE_DEVICES=$1
GROUP=$2
PROMPT=$3
MODEL=$4
NUM_SAMPLES=$5

cnt=0
for model_output in `ls data/model_outputs/NQ301_*.jsonl`; do
  if [ `expr $cnt % 2` == $GROUP ]
  then
    echo -e "${Green}Evaluating $model_output ...${NC}"
    python -m qaeval $model_output --prompt $PROMPT --model $MODEL --do_greedy --num_beams 10 --num_samples $NUM_SAMPLES --batch_size 1
  else
    echo -e "${Green}Evaluating $model_output ...${NC}"
    python -m qaeval $model_output --prompt $PROMPT --model $MODEL --do_greedy --num_beams 10 --num_samples $NUM_SAMPLES --batch_size 1
  fi
  ((cnt++))
done

