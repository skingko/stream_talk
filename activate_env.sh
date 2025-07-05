#!/bin/bash

# Stream-Omni conda环境激活脚本

echo "🐍 激活Stream-Omni conda环境..."

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ conda命令未找到，请确保已安装Anaconda或Miniconda"
    exit 1
fi

# 检查stream_omni环境是否存在
if ! conda env list | grep -q "stream_omni"; then
    echo "❌ stream_omni环境不存在"
    echo "💡 创建环境命令: conda create -n stream_omni python=3.11"
    echo "💡 然后运行: pip install -r requirements.txt"
    exit 1
fi

# 激活环境
echo "✅ 激活stream_omni环境"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stream_omni

# 验证环境
echo "🔍 当前Python环境: $(which python)"
echo "🔍 当前conda环境: $CONDA_DEFAULT_ENV"

# 验证关键包
echo "🔍 验证关键包安装..."
python -c "
try:
    import torch
    import faster_whisper
    import fastapi
    import websockets
    import librosa
    print('✅ 所有核心库验证成功!')
except ImportError as e:
    print(f'❌ 缺少依赖库: {e}')
    print('💡 请运行: pip install -r requirements.txt')
    exit(1)
"

echo "🎉 环境准备完成!"
echo "💡 现在可以运行:"
echo "   - python start_backend.py  (启动后端)"
echo "   - python start_all.py      (启动完整系统)"
