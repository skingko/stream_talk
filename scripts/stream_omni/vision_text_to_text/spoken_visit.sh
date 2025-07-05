#!/bin/bash
unset CUDA_VISIBLE_DEVICES
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


model_path=PATH_TO_STREAMOMNI_CKPT
CKPT=stream_omni
echo "Model path is set to: $model_path"

mkdir -p ./exp/spokenvisit/text_to_text

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python stream_omni/eval/vision_text_to_text/model_vqa_loader_visit.py \
        --model-path ${model_path} \
        --question-file ./playground/SpokenVisIT/VisIT_speech.jsonl \
        --image-folder ./playground/SpokenVisIT/images \
        --answers-file ./exp/spokenvisit/text_to_text/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode stream_omni_llama_3_1 --model-name stream-omni  &
done

wait

output_file=./exp/spokenvisit/text_to_text/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./exp/spokenvisit/text_to_text/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
