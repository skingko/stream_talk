#!/bin/bash
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=CosyVoice/third_party/Matcha-TTS

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


model_path=PATH_TO_STREAMOMNI_CKPT
CKPT=stream_omni
echo "Model path is set to: $model_path"

# download librispeech from https://www.openslr.org/12
# Put all test wavs (i.e., XXX.flac) into ./playground/asr/librispeech/test-clean/wavs and ./playground/asr/librispeech/test-other/wavs

for SPLIT in test-clean test-other; do
    mkdir -p ./exp/asr/librispeech/$SPLIT/$CKPT
    echo $SPLIT >> ./exp/asr/librispeech/$SPLIT/$CKPT/res.txt
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python stream_omni/eval/asr/model_asr.py \
            --model-path ${model_path} \
            --question-file scripts/our/asr/librispeech/$SPLIT/$SPLIT.jsonl\
            --wav-dir scripts/our/asr/librispeech/$SPLIT/wavs \
            --answers-file ./exp/asr/librispeech/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode stream_omni_llama_3_1 --model-name stream-omni &
    done

    wait

    output_file=./exp/asr/librispeech/$SPLIT/$CKPT/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./exp/asr/librispeech/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    python stream_omni/eval/asr/wer.py --file $output_file >> ./exp/asr/librispeech/$SPLIT/$CKPT/res.txt

done

