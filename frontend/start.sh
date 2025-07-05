#!/bin/bash

# Stream-Omni 前端启动脚本

echo "🚀 启动 Stream-Omni 前端开发服务器..."

# 检查是否安装了依赖
if [ ! -d "node_modules" ]; then
    echo "📦 安装依赖包..."
    npm install
fi

# 启动开发服务器
echo "🌐 启动开发服务器 (http://localhost:5173)..."
npm run dev
