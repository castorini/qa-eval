#!/usr/bin/env bash

Green='\033[0;32m'
NC='\033[0m' # No Color

ANNOTATE_FILE=$1

for model_output in `ls data/model_output/NQ_*.jsonl`; do
  echo "${Green}evaluating $model_output ...${NC}"
  python -m qaeval $model_output --annotation $ANNOTATE_FILE
done

model_output="data/model_output/NQ301_text-davinci-003_zeroshot.jsonl"
echo "${Green}evaluating $model_output ...${NC}"
python -m qaeval $model_output --annotation $ANNOTATE_FILE

model_output="data/model_output/NQ301_text-davinci-003_fewshot-n64.jsonl"
echo "${Green}evaluating $model_output ...${NC}"
python -m qaeval $model_output --annotation $ANNOTATE_FILE
