#!/usr/bin/env bash

# scripts/eval_gpt.sh prompts/eval-v0.2-few-shot_chat.txt gpt-3.5-turbo-1106 gpt35t
# scripts/eval_gpt.sh prompts/eval-v0.2-few-shot_chat.txt gpt-4-1106-preview gpt4all

Green='\033[0;32m'
NC='\033[0m' # No Color

PROMPT=$1
MODEL=$2
DEPLOYMENT=$3

for model_output in `ls data/model_outputs/NQ301_*.jsonl`; do
  echo -e "${Green}Evaluating $model_output ...${NC}"
  python -m qaeval $model_output --prompt $PROMPT --model $MODEL --azure --top_p 0.9 --deployment_name $DEPLOYMENT
done
