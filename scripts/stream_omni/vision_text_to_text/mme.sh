#!/bin/bash
unset CUDA_VISIBLE_DEVICES
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


model_path=PATH_TO_STREAMOMNI_CKPT
CKPT=stream_omni
echo "Model path is set to: $model_path"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python stream_omni/eval/vision_text_to_text/model_vqa_loader.py \
        --model-path ${model_path} \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/vision_text_to_text/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode stream_omni_llama_3_1 --model-name stream-omni  &
done

wait

output_file=./playground/data/eval/MME/answers/our_text.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/MME/answers/vision_text_to_text/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


cp -r $output_file $DATA_ROOT/eval/MME/answers

cd $DATA_ROOT/eval/MME 


python convert_answer_to_mme.py --experiment our_text

cd eval_tool

python calculation.py --results_dir answers/our_text >> ./playground/data/eval/MME/answers/vision_text_to_text/$CKPT/res.txt