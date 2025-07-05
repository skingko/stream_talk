#!/bin/bash
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=CosyVoice/third_party/Matcha-TTS

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


model_path=PATH_TO_STREAMOMNI_CKPT
CKPT=stream_omni
echo "Model path is set to: $model_path"

mkdir -p ./exp/spokenqa/web-questions/speech_to_speech/$CKPT

# download llama questions benchmark using ./playground/spokenqa/download.py

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python stream_omni/eval/vision_speech_to_speech/model_sqa_loader_web.py \
        --model-path ${model_path} \
        --data ./playground/spokenqa/speech-web-questions \
        --answers-file ./exp/spokenqa/web-questions/speech_to_speech/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode stream_omni_llama_3_1 --model-name stream-omni &
done

wait

output_file=./exp/spokenqa/web-questions/speech_to_speech/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./exp/spokenqa/web-questions/speech_to_speech/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
python scripts/sqa_acc_speech.py  --file $output_file > ./exp/spokenqa/web-questions/speech_to_speech/$CKPT/res.txt

cat ./exp/spokenqa/web-questions/speech_to_speech/$CKPT/res.txt