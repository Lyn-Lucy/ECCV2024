#!/bin/bash

CHUNKS=2
RATIO="0.125"
MODE="mytome"
SIM_THRESH=0.02
RETAIN_NUM=-1
MAX_PRO=36
TOP_NUM=100

FILE_NAME="$RATIO-$MODE-100-36"

for IDX in {0..1}; do
    CUDA_VISIBLE_DEVICES=$((2*$IDX+4)),$((2*$IDX+5)) python -m llava.eval.model_vqa_science \
        --model-path /home/lzh/llx/models/models--liuhaotian--llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
        --question-file /home/lzh/llx/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
        --image-folder /home/lzh/llx/ScienceQA/data/scienceqa/images/test \
        --answers-file /home/lzh/llx/ScienceQA/res/result-11/$FILE_NAME-chunk_$IDX.jsonl \
        --num-chunks $CHUNKS \
        --merge_mode $MODE\
        --ratio $RATIO \
        --max_pro $MAX_PRO \
        --topnum $TOP_NUM \
        --chunk-idx $IDX \
        --conv-mode llava_v1 &
done

# CHUNKS=1
# RATIO="0.125"
# MODE="mytome"
# SIM_THRESH=0.02
# RETAIN_NUM=-1
# MAX_PRO=36
# TOP_NUM=100

# FILE_NAME="0.125-mytome-100-36"

# for IDX in {0}; do
#     CUDA_VISIBLE_DEVICES=0,1 python -m llava.eval.model_vqa_science \
#         --model-path /home/lzh/llx/models/models--liuhaotian--llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
#         --question-file /home/lzh/llx/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
#         --image-folder /home/lzh/llx/ScienceQA/data/scienceqa/images/test \
#         --answers-file /home/lzh/llx/ScienceQA/res/result-11/$FILE_NAME.jsonl \
#         --num-chunks $CHUNKS \
#         --merge_mode $MODE\
#         --ratio $RATIO \
#         --max_pro $MAX_PRO \
#         --topnum $TOP_NUM \
#         --chunk-idx 0 \
#         --conv-mode llava_v1 &
# done
        # --answers-file /home/lzh/llx/ScienceQA/res/result-3/T8-$MODE-$RATIO-test_llava-13b-chunk$CHUNKS_$IDX.jsonl \
        # --merge_mode $MODE \
        # --ratio $RATIO \
        # --max_pro 50 \

# CHUNKS=8
# for IDX in {0..7}; do
#     CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_vqa_science \
#         --model-path liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
#         --question-file ~/haotian/datasets/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
#         --image-folder ~/haotian/datasets/ScienceQA/data/scienceqa/images/test \
#         --answers-file ./test_llava-13b-chunk$CHUNKS_$IDX.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --conv-mode llava_v1 &
# done


    # CUDA_VISIBLE_DEVICES=6,7 python -m llava.eval.model_vqa_science \
    #     --model-path /home/lzh/llx/models/models--liuhaotian--llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
    #     --question-file /home/lzh/llx/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
    #     --image-folder /home/lzh/llx/ScienceQA/data/scienceqa/images/test \
    #     --answers-file /home/lzh/llx/ScienceQA/res/result-10/test.jsonl \
    #     --num-chunks 1 \
    #     --merge_mode "nomerge" \
    #     --ratio 0.125 \
    #     --max_pro 50 \
    #     --topnum 100 \
    #     --chunk-idx 0 \
    #     --conv-mode llava_v1