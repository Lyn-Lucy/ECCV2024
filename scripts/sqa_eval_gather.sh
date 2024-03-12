#!/bin/bash

CHUNKS=2
RATIO="0.125"
MODE="tome"
res=4

FILE_NAME="0.125-mytome-100-36"

output_file="/home/lzh/llx/ScienceQA/res/result-11/$FILE_NAME.jsonl"
# Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
for idx in $(seq 0 $((CHUNKS-1))); do
  cat "/home/lzh/llx/ScienceQA/res/result-11/$FILE_NAME-chunk_${idx}.jsonl" >> "$output_file"
done

python ../llava/eval/eval_science_qa.py \
    --base-dir /home/lzh/llx/ScienceQA/data/scienceqa \
    --result-file /home/lzh/llx/ScienceQA/res/result-11/$FILE_NAME.jsonl \
    --output-file /home/lzh/llx/ScienceQA/res/result-11/$FILE_NAME-output.json \
    --output-result /home/lzh/llx/ScienceQA/res/result-11/$FILE_NAME-result.json \


# CHUNKS=4
# output_file="/home/lzh/llx/ScienceQA/results/test1_llava-13b.jsonl"

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for idx in $(seq 0 $((CHUNKS-1))); do
#   cat "/home/lzh/llx/ScienceQA/results/test1_llava-13b-chunk${idx}.jsonl" >> "$output_file"
# done

# python /home/lzh/llx/LLaVA/llava/eval/eval_science_qa.py \
#     --0.0625-dir /home/lzh/llx/ScienceQA/data/scienceqa \
#     --result-file /home/lzh/llx/ScienceQA/results/test1_llava-13b.jsonl \
#     --output-file /home/lzh/llx/ScienceQA/results/test1_llava-13b_output.json \
#     --output-result /home/lzh/llx/ScienceQA/results/test1_llava-13b_result.json
