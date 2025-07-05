#!/usr/bin/env python3
"""
第三方库路径配置
统一管理第三方库的路径，便于导入和使用
"""

import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 第三方库根目录
THIRD_PARTY_ROOT = PROJECT_ROOT / "third-party"

# 各个第三方库的路径
FISH_SPEECH_PATH = THIRD_PARTY_ROOT / "fish-speech"
TEN_FRAMEWORK_PATH = THIRD_PARTY_ROOT / "ten-framework"
TEN_VAD_PATH = THIRD_PARTY_ROOT / "ten-vad"
TEN_TURN_DETECTION_PATH = THIRD_PARTY_ROOT / "ten-turn-detection"

def add_third_party_paths():
    """将第三方库路径添加到Python路径中"""
    paths_to_add = [
        str(FISH_SPEECH_PATH),
        str(TEN_FRAMEWORK_PATH),
        str(TEN_VAD_PATH),
        str(TEN_TURN_DETECTION_PATH),
    ]
    
    for path in paths_to_add:
        if Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)
            print(f"✅ 已添加第三方库路径: {path}")

def get_fish_speech_path() -> Path:
    """获取Fish Speech路径"""
    return FISH_SPEECH_PATH

def get_ten_framework_path() -> Path:
    """获取TEN Framework路径"""
    return TEN_FRAMEWORK_PATH

def get_ten_vad_path() -> Path:
    """获取TEN VAD路径"""
    return TEN_VAD_PATH

def get_ten_turn_detection_path() -> Path:
    """获取TEN Turn Detection路径"""
    return TEN_TURN_DETECTION_PATH

def check_third_party_availability():
    """检查第三方库是否可用"""
    libraries = {
        "Fish Speech": FISH_SPEECH_PATH,
        "TEN Framework": TEN_FRAMEWORK_PATH,
        "TEN VAD": TEN_VAD_PATH,
        "TEN Turn Detection": TEN_TURN_DETECTION_PATH,
    }
    
    available = {}
    for name, path in libraries.items():
        available[name] = path.exists()
        status = "✅" if available[name] else "❌"
        print(f"{status} {name}: {path}")
    
    return available

if __name__ == "__main__":
    print("🔍 检查第三方库可用性:")
    check_third_party_availability()
    
    print("\n📦 添加第三方库路径:")
    add_third_party_paths()
