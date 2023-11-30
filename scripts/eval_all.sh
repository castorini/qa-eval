#!/usr/bin/env bash

Green='\033[0;32m'
NC='\033[0m' # No Color

ANNOTATE_FILE=$1

for model_output in `ls data/model_outputs/NQ_*.jsonl`; do
  echo -e "${Green}Evaluating $model_output ...${NC}"
  python -m qaeval $model_output --annotation $ANNOTATE_FILE
done

model_output="data/model_outputs/NQ301_text-davinci-003_zeroshot.jsonl"
echo -e "${Green}Evaluating $model_output ...${NC}"
python -m qaeval $model_output --annotation $ANNOTATE_FILE

model_output="data/model_outputs/NQ301_text-davinci-003_fewshot-n64.jsonl"
echo -e "${Green}Evaluating $model_output ...${NC}"
python -m qaeval $model_output --annotation $ANNOTATE_FILE
