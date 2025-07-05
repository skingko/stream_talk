#!/bin/bash
unset CUDA_VISIBLE_DEVICES
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


model_path=PATH_TO_STREAMOMNI_CKPT
CKPT=stream_omni
echo "Model path is set to: $model_path"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python stream_omni/eval/vision_text_to_text/model_vqa_science.py \
        --model-path ${model_path} \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/vision_text_to_text/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode stream_omni_llama_3_1 --model-name stream-omni  &
done

wait

output_file=./playground/data/eval/scienceqa/answers/vision_text_to_text/$CKPT.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/scienceqa/answers/vision_text_to_text/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_science_qa.py \
    --base-dir $DATA_ROOT/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/vision_text_to_text/$CKPT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/vision_text_to_text/$CKPT/merge_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/vision_text_to_text/$CKPT/merge_result.jsonl