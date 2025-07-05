#!/bin/bash
unset CUDA_VISIBLE_DEVICES
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


model_path=PATH_TO_STREAMOMNI_CKPT
CKPT=stream_omni
echo "Model path is set to: $model_path"

SPLIT="llava_gqa_testdev_balanced"
GQA_DATA=./playground/data
GQADIR=./playground/data/eval/gqa/data

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python stream_omni/eval/vision_text_to_text/model_vqa_loader.py \
        --model-path ${model_path} \
        --question-file $GQA_DATA/eval/gqa/$SPLIT.jsonl \
        --image-folder $GQA_DATA/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode stream_omni_llama_3_1 --model-name stream-omni &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQA_DATA/eval/gqa/data/testdev_balanced_predictions.json

cd $GQA_DATA/eval/gqa/data
python eval/eval.py --tier testdev_balanced > ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/res.txt

cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/res.txt
