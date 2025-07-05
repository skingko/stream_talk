#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# CKPT=$1
# CKPT=$(basename "$CKPT")

# if [ -n "$2" ]; then
#     model_path="checkpoints/${CKPT}/checkpoint-${2}"
#     CKPT=$CKPT:update-${2}
# else
#     model_path="checkpoints/${CKPT}"
# fi
model_path=./checkpoints/llavanext-siglip-so400m-patch14-384-Meta-Llama-3.1-8B-Instruct-mlp2x_gelu-fineturn_next+ov

CKPT=$(basename "$model_path")+".square"
echo "Model path is set to: $model_path"

SPLIT="mmbench_dev_20230712"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./stream_omni/eval/vision_text_to_text/model_vqa_mmbench.py \
        --model-path ${model_path} \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode stream_omni_llama_3_1 --model-name stream-omni &
done

wait

output_file=./playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
