#!/usr/bin/env python3
"""
Stream-Omni 完整系统启动脚本
同时启动前端和后端服务
"""

import os
import sys
import subprocess
import logging
import time
import signal
from pathlib import Path
from threading import Thread

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class StreamOmniLauncher:
    """Stream-Omni系统启动器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backend_process = None
        self.frontend_process = None
        self.running = True
    
    def check_conda_env(self):
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
    
    def check_requirements(self):
        """检查系统要求"""
        logger.info("🔍 检查系统要求...")
        
        # 检查后端脚本
        backend_script = self.project_root / "simple_voice_server.py"
        if not backend_script.exists():
            logger.error("❌ 后端脚本不存在: simple_voice_server.py")
            return False
        
        # 检查前端目录
        frontend_dir = self.project_root / "frontend"
        if not frontend_dir.exists():
            logger.error("❌ 前端目录不存在: frontend/")
            return False
        
        # 检查package.json
        package_json = frontend_dir / "package.json"
        if not package_json.exists():
            logger.error("❌ 前端配置文件不存在: frontend/package.json")
            return False
        
        # 检查conda环境
        if not self.check_conda_env():
            logger.error("❌ conda环境 'stream_omni' 不存在")
            logger.info("💡 创建环境命令: conda create -n stream_omni python=3.11")
            return False
        
        logger.info("✅ 系统要求检查完成")
        return True
    
    def start_backend(self):
        """启动后端服务"""
        try:
            logger.info("🚀 启动后端服务...")
            
            # 构建后端启动命令
            cmd = [
                "bash", "-c",
                "source ~/miniconda3/etc/profile.d/conda.sh && "
                "conda activate stream_omni && "
                "python -c \""
                "import uvicorn; "
                "from simple_voice_server import voice_server; "
                "uvicorn.run(voice_server.app, host='0.0.0.0', port=8002, log_level='info', reload=False)"
                "\""
            ]
            
            # 启动后端进程
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            logger.info("✅ 后端服务启动中...")
            return True
            
        except Exception as e:
            logger.error(f"❌ 后端服务启动失败: {e}")
            return False
    
    def start_frontend(self):
        """启动前端服务"""
        try:
            logger.info("🚀 启动前端服务...")
            
            frontend_dir = self.project_root / "frontend"
            
            # 检查是否需要安装依赖
            node_modules = frontend_dir / "node_modules"
            if not node_modules.exists():
                logger.info("📦 安装前端依赖...")
                install_result = subprocess.run(
                    ["npm", "install"],
                    cwd=frontend_dir,
                    capture_output=True,
                    text=True
                )
                if install_result.returncode != 0:
                    logger.error(f"❌ 前端依赖安装失败: {install_result.stderr}")
                    return False
            
            # 启动前端开发服务器
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            logger.info("✅ 前端服务启动中...")
            return True
            
        except Exception as e:
            logger.error(f"❌ 前端服务启动失败: {e}")
            return False
    
    def monitor_processes(self):
        """监控进程状态"""
        def monitor_backend():
            if self.backend_process:
                for line in iter(self.backend_process.stdout.readline, ''):
                    if self.running:
                        print(f"[后端] {line.strip()}")
                    else:
                        break
        
        def monitor_frontend():
            if self.frontend_process:
                for line in iter(self.frontend_process.stdout.readline, ''):
                    if self.running:
                        print(f"[前端] {line.strip()}")
                    else:
                        break
        
        # 启动监控线程
        if self.backend_process:
            Thread(target=monitor_backend, daemon=True).start()
        
        if self.frontend_process:
            Thread(target=monitor_frontend, daemon=True).start()
    
    def stop_services(self):
        """停止所有服务"""
        logger.info("🛑 停止所有服务...")
        self.running = False
        
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                logger.info("✅ 后端服务已停止")
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                logger.info("🔪 强制停止后端服务")
        
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                logger.info("✅ 前端服务已停止")
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                logger.info("🔪 强制停止前端服务")
    
    def run(self):
        """运行完整系统"""
        try:
            print("="*60)
            print("🎯 Stream-Omni 完整系统启动器")
            print("="*60)
            
            # 检查要求
            if not self.check_requirements():
                sys.exit(1)
            
            # 启动后端
            if not self.start_backend():
                sys.exit(1)
            
            # 等待后端启动
            time.sleep(3)
            
            # 启动前端
            if not self.start_frontend():
                self.stop_services()
                sys.exit(1)
            
            # 等待前端启动
            time.sleep(2)
            
            logger.info("="*60)
            logger.info("🎉 Stream-Omni 系统启动完成!")
            logger.info("📡 后端API: http://localhost:8002")
            logger.info("🌐 前端界面: http://localhost:5174")
            logger.info("🎤 WebSocket: ws://localhost:8002/ws/voice")
            logger.info("💊 健康检查: http://localhost:8002/health")
            logger.info("="*60)
            logger.info("按 Ctrl+C 停止所有服务")
            
            # 监控进程
            self.monitor_processes()
            
            # 等待用户中断
            while self.running:
                time.sleep(1)
                
                # 检查进程是否还在运行
                if self.backend_process and self.backend_process.poll() is not None:
                    logger.error("❌ 后端服务意外退出")
                    break
                
                if self.frontend_process and self.frontend_process.poll() is not None:
                    logger.error("❌ 前端服务意外退出")
                    break
            
        except KeyboardInterrupt:
            logger.info("⏹️ 用户中断服务")
        except Exception as e:
            logger.error(f"❌ 系统运行失败: {e}")
        finally:
            self.stop_services()

def signal_handler(signum, frame):
    """信号处理器"""
    logger.info("🛑 收到停止信号")
    sys.exit(0)

def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动系统
    launcher = StreamOmniLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
