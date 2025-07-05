#!/usr/bin/env python3
"""
Stream-Omni 后端启动脚本
使用conda环境启动后端服务
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def check_conda_env():
    """检查conda环境是否存在"""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        return "stream_omni" in result.stdout
    except Exception as e:
        logger.error(f"❌ 检查conda环境失败: {e}")
        return False

def check_requirements():
    """检查后端要求"""
    logger.info("🔍 检查后端要求...")

    project_root = Path(__file__).parent

    # 检查后端脚本
    backend_script = project_root / "simple_voice_server.py"
    if not backend_script.exists():
        logger.error("❌ 后端脚本不存在: simple_voice_server.py")
        return False

    # 检查conda环境
    if not check_conda_env():
        logger.error("❌ conda环境 'stream_omni' 不存在")
        logger.info("💡 创建环境命令: conda create -n stream_omni python=3.11")
        return False

    logger.info("✅ 后端要求检查完成")
    return True

def start_with_conda():
    """使用conda环境启动后端服务"""
    try:
        logger.info("🚀 启动Stream-Omni后端服务...")
        logger.info("🐍 使用conda环境: stream_omni")
        logger.info("⚡ TTS引擎: Spark-TTS")
        logger.info("📡 后端API: http://localhost:8002")
        logger.info("🎤 WebSocket: ws://localhost:8002/ws/voice")
        logger.info("💊 健康检查: http://localhost:8002/health")
        logger.info("="*60)

        # 构建启动命令
        cmd = [
            "bash", "-c",
            "source ~/miniconda3/etc/profile.d/conda.sh && "
            "conda activate stream_omni && "
            "cd /Users/apple/Documents/AI智能代码/Livevibe/streem-omni && "
            "python simple_voice_server.py"
        ]

        # 启动服务
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode == 0

    except KeyboardInterrupt:
        logger.info("⏹️ 用户中断后端服务")
        return True
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        return False

def start_direct():
    """直接启动（如果已在正确环境中）"""
    try:
        import uvicorn
        from simple_voice_server import voice_server

        logger.info("🚀 启动Stream-Omni后端服务...")
        logger.info("🐍 使用当前Python环境")
        logger.info("⚡ TTS引擎: Spark-TTS")
        logger.info("📡 后端API: http://localhost:8002")
        logger.info("🎤 WebSocket: ws://localhost:8002/ws/voice")
        logger.info("💊 健康检查: http://localhost:8002/health")
        logger.info("="*60)

        # 启动FastAPI服务器
        uvicorn.run(
            voice_server.app,
            host="0.0.0.0",
            port=8002,
            log_level="info",
            reload=False
        )
        return True

    except ImportError as e:
        logger.error(f"❌ 导入失败，请确保在正确的conda环境中: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("⏹️ 用户中断后端服务")
        return True
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        return False

def main():
    """主启动函数"""
    print("="*60)
    print("🎯 Stream-Omni 后端服务启动器")
    print("⚡ TTS引擎: Spark-TTS")
    print("="*60)

    # 检查要求
    if not check_requirements():
        sys.exit(1)

    # 检查是否已在conda环境中
    current_env = os.environ.get('CONDA_DEFAULT_ENV')

    if current_env == 'stream_omni':
        logger.info("✅ 已在stream_omni环境中")
        success = start_direct()
    else:
        logger.info("🔄 切换到stream_omni环境")
        success = start_with_conda()

    if success:
        logger.info("✅ 后端服务正常退出")
    else:
        logger.error("❌ 后端服务异常退出")
        sys.exit(1)

if __name__ == "__main__":
    main()
