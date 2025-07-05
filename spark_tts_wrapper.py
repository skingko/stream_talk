#!/usr/bin/env python3
"""
Spark-TTS包装器
基于mlx-audio实现，支持流式语音合成和参考音频voice cloning
"""

import os
import sys
import logging
import re
import asyncio
import threading
import gc
import time
from typing import Optional, Generator, Dict, Any, List
import numpy as np
import soundfile as sf
from pathlib import Path

# 导入MLX用于GPU资源管理
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)

# 全局锁，确保模型访问的线程安全
_model_lock = threading.Lock()
_global_model_instance = None

class SparkTTSWrapper:
    """Spark-TTS包装器，基于mlx-audio实现，支持流式语音合成和参考音频voice cloning"""

    def __init__(self, model_path: str = None):
        """
        初始化Spark-TTS包装器

        Args:
            model_path: Spark-TTS模型路径
        """
        # 尝试多个模型路径，优先使用本地修复的模型
        self.model_candidates = [
            model_path or "models/spark-tts/Spark-TTS-0.5B-6bit",  # 本地修复的6bit模型
            "mlx-community/Spark-TTS-0.5B",  # 远程非量化模型
            "SparkAudio/Spark-TTS-0.5B",  # 官方非量化模型
        ]
        self.model_path = None
        self.model_loaded = False
        self.sample_rate = 16000  # 与直接调用保持一致的采样率

        # 温柔女声配置
        self.voice_settings = {
            "gender": "female",
            "pitch": -2.0,  # 降低音调，更温柔
            "speed": 0.8,   # 调整为更自然的语速
            "temperature": 0.7,  # 适中的随机性
            "voice": "af_heart"  # 温柔女声
        }

        # 默认参考音频
        self.default_reference_audio = "assets/音频样本1.-温婉女声wav.wav"
        self.reference_text = "我是一个快乐的大语言模型生活助手，能够流畅的输出语音表达自己的想法，我每天都很开心并努力工作，希望你以后能够做的比我还好哟。"

        # Spark-TTS支持的速度映射 - 只能使用固定的5个值
        # Spark-TTS内部SPEED_MAP: {0.0: 'very_low', 0.5: 'low', 1.0: 'moderate', 1.5: 'high', 2.0: 'very_high'}
        # 我们将用户输入的速度映射到这5个支持的值
        self.speed_map = {
            0.0: 0.0,   # 极慢 -> very_low
            0.5: 0.5,   # 慢 -> low
            1.0: 0.5,   # 正常 -> low (更自然的语速)
            1.5: 1.0,   # 快 -> moderate
            2.0: 1.5    # 极快 -> high
        }

        # 持久化模型组件
        self.persistent_model = None
        self.model_preloaded = False
        self.model_warmup_done = False

        # 初始化模型
        self._initialize_model()

        # 预加载模型以实现持久化推理
        self._preload_persistent_model()

        logger.info(f"✅ Spark-TTS包装器初始化完成")

    def _map_speed(self, speed: float) -> float:
        """将任意速度值映射到Spark-TTS支持的速度值"""
        supported_speeds = list(self.speed_map.keys())

        # 如果速度值已经在支持列表中，直接返回映射值
        if speed in supported_speeds:
            mapped_value = self.speed_map[speed]
            logger.debug(f"🎵 速度映射: {speed} -> {mapped_value}")
            return mapped_value

        # 找到最接近的支持速度值
        closest_speed = min(supported_speeds, key=lambda x: abs(x - speed))
        mapped_value = self.speed_map[closest_speed]
        logger.debug(f"🎵 速度映射: {speed} -> {closest_speed} -> {mapped_value}")
        return mapped_value

    def _initialize_model(self):
        """初始化Spark-TTS模型"""
        try:
            # 导入mlx-audio模块
            from mlx_audio.tts.generate import generate_audio
            from mlx_audio.tts.utils import load_model, get_model_path

            self.generate_audio = generate_audio
            self.load_model = load_model
            self.get_model_path = get_model_path

            # 尝试找到可用的模型
            for model_candidate in self.model_candidates:
                try:
                    logger.info(f"🔍 尝试模型: {model_candidate}")
                    # 尝试获取模型路径（这会触发下载或验证）
                    model_path = get_model_path(model_candidate)
                    self.model_path = model_candidate
                    logger.info(f"✅ 找到可用模型: {model_candidate}")
                    break
                except Exception as model_error:
                    logger.warning(f"⚠️ 模型 {model_candidate} 不可用: {model_error}")
                    continue

            if self.model_path:
                logger.info(f"📦 使用Spark-TTS模型: {self.model_path}")
                self.model_loaded = True
                logger.info("✅ Spark-TTS模型初始化完成")
            else:
                raise RuntimeError("所有模型候选都不可用")

        except Exception as e:
            logger.error(f"❌ Spark-TTS模型初始化失败: {e}")
            raise RuntimeError(f"Spark-TTS模型初始化失败: {e}")

    def _cleanup_gpu_resources(self):
        """清理GPU资源，防止Metal错误"""
        try:
            if MLX_AVAILABLE:
                # 使用新的API清理GPU缓存
                mx.clear_cache()
                logger.debug("🧹 GPU缓存已清理")

            # Python垃圾回收
            gc.collect()

            # 更长的等待时间，让GPU资源完全释放
            time.sleep(0.5)

        except Exception as e:
            logger.warning(f"⚠️ GPU资源清理失败: {e}")

    def _safe_model_access(self, operation_func, *args, **kwargs):
        """线程安全的模型访问"""
        global _global_model_instance, _model_lock

        with _model_lock:
            try:
                # 清理之前的GPU资源
                self._cleanup_gpu_resources()

                # 执行操作
                result = operation_func(*args, **kwargs)

                # 操作完成后再次清理
                self._cleanup_gpu_resources()

                return result

            except Exception as e:
                logger.error(f"❌ 安全模型访问失败: {e}")
                # 发生错误时强制清理
                self._cleanup_gpu_resources()
                raise e

    def _preload_persistent_model(self):
        """预加载持久化模型，避免每次推理时重新加载"""
        try:
            if not self.model_loaded:
                logger.warning("⚠️ 模型未初始化，跳过预加载")
                return

            logger.info("🔄 开始预加载持久化模型...")

            # 清理GPU资源后再加载
            self._cleanup_gpu_resources()

            # 使用mlx-audio的load_model预加载模型，使用strict=False忽略量化参数不匹配
            self.persistent_model = self.load_model(model_path=self.model_path, strict=False)
            self.model_preloaded = True

            logger.info("✅ 持久化模型预加载成功")

            # 执行模型预热
            self._warmup_persistent_model()

        except Exception as e:
            logger.warning(f"⚠️ 持久化模型预加载失败，将使用标准方式: {e}")
            self.persistent_model = None
            self.model_preloaded = False
            # 预加载失败时也要清理资源
            self._cleanup_gpu_resources()

    def _warmup_persistent_model(self):
        """预热持久化模型，减少首次推理延迟"""
        try:
            if not self.model_preloaded:
                return

            logger.info("🔥 开始模型预热...")

            # 使用简短文本进行预热推理
            warmup_text = "你好"
            warmup_audio = self._generate_with_persistent_model(
                text=warmup_text,
                speed=1.0,
                reference_audio=None,
                warmup=True
            )

            if warmup_audio is not None:
                self.model_warmup_done = True
                logger.info("✅ 模型预热完成")
            else:
                logger.warning("⚠️ 模型预热失败，但继续运行")

        except Exception as e:
            logger.warning(f"⚠️ 模型预热失败: {e}")
            self.model_warmup_done = False

    def _generate_with_persistent_model(self, text: str, speed: float, reference_audio: str, warmup: bool = False) -> Optional[np.ndarray]:
        """使用持久化模型进行音频生成"""
        try:
            if not self.model_preloaded or self.persistent_model is None:
                # 如果持久化模型不可用，回退到标准方式
                return None

            # 创建临时输出目录
            output_dir = Path("temp_audio")
            output_dir.mkdir(exist_ok=True)

            # 生成唯一的文件前缀
            import time
            file_prefix = f"temp_audio/persistent_{int(time.time() * 1000)}"
            if warmup:
                file_prefix = f"temp_audio/warmup_{int(time.time() * 1000)}"

            # 使用与直接调用完全相同的参数逻辑
            # 不进行任何速度映射或组合，直接使用传入的速度
            final_speed = speed  # 直接使用传入的速度，不做任何修改
            logger.info(f"🎵 使用直接调用逻辑: 传入速度={speed}, 最终速度={final_speed}")

            # 使用与直接调用完全相同的参数
            self.generate_audio(
                text=text,
                model_path=self.model_path,
                voice=None,
                speed=final_speed,  # 直接使用传入的速度
                lang_code='a',
                ref_audio=reference_audio if reference_audio and os.path.exists(reference_audio) else None,
                ref_text=self.reference_text if reference_audio else None,
                file_prefix=file_prefix,
                audio_format="wav",
                join_audio=True,
                play=False,
                verbose=True if warmup else False
            )

            # 读取生成的音频文件
            audio_file = Path(f"{file_prefix}.wav")
            if audio_file.exists():
                audio_data, _ = sf.read(str(audio_file))

                # 清理临时文件
                audio_file.unlink()

                # 确保音频数据格式正确
                if isinstance(audio_data, np.ndarray):
                    if audio_data.ndim == 2:  # 立体声转单声道
                        audio_data = np.mean(audio_data, axis=1)

                    if not warmup:
                        logger.info(f"✅ 持久化模型生成音频: {len(audio_data)} 样本")

                    return audio_data.astype(np.float32)

            return None

        except Exception as e:
            if not warmup:
                logger.error(f"❌ 持久化模型推理失败: {e}")
            return None
    
    def synthesize_stream(self,
                         text: str,
                         emotion: str = "neutral",
                         speed: float = 1.0,
                         reference_audio: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        流式语音合成 - 使用参考音频方法

        Args:
            text: 要合成的文本
            emotion: 情感标记
            speed: 语速
            reference_audio: 参考音频文件路径

        Yields:
            Dict: 包含音频数据的字典
        """
        try:
            if not self.model_loaded:
                raise RuntimeError("Spark-TTS模型未加载")

            # 使用默认的温婉女声样本
            if not reference_audio:
                reference_audio = self.default_reference_audio

            yield from self._synthesize_with_reference(text, emotion, speed, reference_audio)

        except Exception as e:
            logger.error(f"❌ Spark-TTS合成失败: {e}")
            raise RuntimeError(f"TTS合成失败: {e}")

    def _synthesize_with_reference(self, text: str, emotion: str, speed: float, reference_audio: str) -> Generator[Dict[str, Any], None, None]:
        """使用参考音频进行语音合成 - 基于mlx-audio实现"""
        try:
            # 处理情感和语气标记
            processed_text = self._process_emotion_markers(text, emotion)
            
            logger.info(f"🎵 开始Spark-TTS合成: {processed_text[:50]}...")
            
            # 实现句级切分+分组输出的流式音频生成
            sentences = self._split_text_to_sentences(processed_text)
            sentence_groups = self._group_sentences_for_output(sentences)

            logger.info(f"🎵 句子分组: 共{len(sentences)}个句子，分为{len(sentence_groups)}组")
            if sentence_groups:
                logger.info(f"🎵 第一组: {len(sentence_groups[0])}个句子，后续每组1个句子")

            for group_index, sentence_group in enumerate(sentence_groups):
                # 合并当前组的句子
                combined_text = "".join(sentence_group)
                if not combined_text.strip():
                    continue

                logger.debug(f"🎶 合成第{group_index+1}组: {combined_text[:50]}...")

                # 使用mlx-audio生成音频
                audio_data = self._generate_sentence_audio(combined_text, speed, reference_audio)

                if audio_data is not None:
                    yield {
                        "audio": audio_data,
                        "sample_rate": self.sample_rate,
                        "is_final": group_index == len(sentence_groups) - 1,
                        "group_size": len(sentence_group),
                        "group_index": group_index
                    }

            logger.info(f"✅ Spark-TTS合成完成，共{len(sentence_groups)}个音频组")

        except Exception as e:
            logger.error(f"❌ Spark-TTS合成失败: {e}")
            raise RuntimeError(f"TTS合成失败: {e}")

    def _split_text_to_sentences(self, text: str) -> List[str]:
        """将文本切分为断句，用于流式处理（更细粒度的分割）"""
        # 移除情感标记，避免影响分割
        text = re.sub(r'\([^)]*\)', '', text)

        # 更细粒度的断句分割，包括逗号、分号等
        # 中文断句：句号、感叹号、问号、逗号、分号、冒号
        sentences = re.split(r'(?<=[。！？，；：])', text)

        # 英文断句：句号、感叹号、问号、逗号、分号、冒号
        sentences = [s for sentence in sentences for s in re.split(r'(?<=[.!?,:;])\s+', sentence)]

        # 过滤空句子和只有标点的句子
        sentences = [s.strip() for s in sentences if s.strip() and not re.match(r'^[。！？，；：.!?,:;\s]*$', s.strip())]

        return sentences

    def _group_sentences_for_output(self, sentences: List[str]) -> List[List[str]]:
        """将断句分组用于输出：第一组3个断句，之后每组1个断句"""
        if not sentences:
            return []

        groups = []

        # 第一组：前3个断句（如果有的话）
        first_group_size = min(3, len(sentences))
        if first_group_size > 0:
            groups.append(sentences[:first_group_size])

        # 后续组：每组1个断句
        for i in range(first_group_size, len(sentences)):
            groups.append([sentences[i]])

        return groups

    def _generate_sentence_audio(self, text: str, speed: float, reference_audio: str) -> Optional[np.ndarray]:
        """生成单个句子的音频 - 优先使用持久化模型"""
        try:
            # 添加更长延迟以避免Metal错误
            time.sleep(1.0)

            # 清理GPU资源
            self._cleanup_gpu_resources()

            # 优先尝试使用持久化模型
            if self.model_preloaded and self.persistent_model is not None:
                audio_data = self._generate_with_persistent_model(text, speed, reference_audio)
                if audio_data is not None:
                    return audio_data
                else:
                    logger.warning("⚠️ 持久化模型推理失败，回退到标准方式")

            # 回退到标准方式
            return self._generate_with_standard_method(text, speed, reference_audio)

        except Exception as e:
            logger.error(f"❌ 句子音频生成失败: {e}")
            # 发生错误时清理GPU资源
            self._cleanup_gpu_resources()
            return None

    def _generate_with_standard_method(self, text: str, speed: float, reference_audio: str) -> Optional[np.ndarray]:
        """使用标准方法生成音频（回退方案）"""
        try:
            # 创建临时输出目录
            output_dir = Path("temp_audio")
            output_dir.mkdir(exist_ok=True)

            # 配置Spark-TTS参数 - 使用温柔女声设置
            model_path = self.model_path

            # 生成唯一的文件前缀
            import time
            file_prefix = f"temp_audio/standard_{int(time.time() * 1000)}"

            # 使用与直接调用完全相同的参数逻辑
            final_speed = speed  # 直接使用传入的速度，不做任何修改
            logger.info(f"🎵 标准方法使用直接调用逻辑: 传入速度={speed}, 最终速度={final_speed}")

            # 使用正确的API参数进行音频生成
            def _standard_generate_operation():
                # 使用与直接调用完全相同的参数
                self.generate_audio(
                    text=text,
                    model_path=model_path,
                    voice=None,
                    speed=final_speed,  # 直接使用传入的速度
                    lang_code='a',
                    ref_audio=reference_audio if reference_audio and os.path.exists(reference_audio) else None,
                    ref_text=self.reference_text if reference_audio else None,
                    file_prefix=file_prefix,
                    audio_format="wav",
                    join_audio=True,
                    play=False,
                    verbose=False
                )
                return file_prefix

            # 使用线程安全的模型访问
            file_prefix = self._safe_model_access(_standard_generate_operation)

            # 读取生成的音频文件
            audio_file = Path(f"{file_prefix}.wav")
            if audio_file.exists():
                audio_data, _ = sf.read(str(audio_file))

                # 清理临时文件
                audio_file.unlink()

                # 确保音频数据格式正确
                if isinstance(audio_data, np.ndarray):
                    if audio_data.ndim == 2:  # 立体声转单声道
                        audio_data = np.mean(audio_data, axis=1)
                    logger.info(f"✅ 标准方法生成音频: {len(audio_data)} 样本")
                    return audio_data.astype(np.float32)

            return None

        except Exception as e:
            logger.error(f"❌ 标准方法音频生成失败: {e}")
            return None



    def _process_emotion_markers(self, text: str, emotion: str) -> str:
        """处理情感标记，Spark-TTS不支持情感标记，直接移除"""
        try:
            # 移除所有情感标记，Spark-TTS不支持
            import re
            text = re.sub(r'\([^)]*\)', '', text).strip()

            # 清理多余的空格
            text = re.sub(r'\s+', ' ', text).strip()

            logger.debug(f"🧹 移除情感标记后: {text}")
            return text

        except Exception as e:
            logger.error(f"❌ 处理情感标记失败: {e}")
            return text

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "model_loaded": self.is_model_loaded(),
            "sample_rate": self.sample_rate,
            "device": "mps" if hasattr(self, 'device') else "auto"
        }

    def configure_voice_settings(self, gender: str = "female", pitch: float = -2.0, speed: float = 1.0):
        """配置温柔女声设置"""
        self.voice_settings.update({
            "gender": gender,
            "pitch": pitch,
            "speed": speed
        })
        logger.info(f"🎤 配置语音设置: gender={gender}, pitch={pitch}, speed={speed}")

    def synthesize_stream_realtime(self,
                                  text: str,
                                  emotion: str = "neutral",
                                  speed: float = 1.0,
                                  reference_audio: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        真正的实时流式语音合成 - 基于句级切分+并行处理

        Args:
            text: 要合成的文本
            emotion: 情感标记
            speed: 语速
            reference_audio: 参考音频文件路径

        Yields:
            Dict: 包含音频数据的字典
        """
        try:
            if not self.model_loaded:
                raise RuntimeError("Spark-TTS模型未加载")

            # 使用默认的温婉女声样本
            if not reference_audio:
                reference_audio = "assets/音频样本1.-温婉女声wav.wav"

            # 处理情感和语气标记
            processed_text = self._process_emotion_markers(text, emotion)

            logger.info(f"🎵 开始Spark-TTS实时流式合成: {processed_text[:50]}...")

            # 实现句级切分+并行推理的流式音频生成
            sentences = self._split_text_to_sentences(processed_text)

            # 使用串行处理避免GPU冲突，禁用并发
            logger.info("🔄 使用串行处理避免Metal错误...")

            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    try:
                        # 串行生成每个句子的音频
                        audio_data = self._generate_sentence_audio_optimized(sentence, speed, reference_audio)
                        if audio_data is not None:
                            yield {
                                "audio": audio_data,
                                "sample_rate": self.sample_rate,
                                "is_final": i == len(sentences) - 1,
                                "sentence_index": i,
                                "sentence_text": sentence[:30] + "..." if len(sentence) > 30 else sentence
                            }
                        else:
                            logger.warning(f"⚠️ 句子 {i + 1} 生成失败")
                    except Exception as e:
                        logger.error(f"❌ 句子 {i + 1} 处理异常: {e}")
                        # 发生错误时清理GPU资源
                        self._cleanup_gpu_resources()
                        continue

            logger.info(f"✅ Spark-TTS实时流式合成完成，共{len(sentences)}个句子")

        except Exception as e:
            logger.error(f"❌ Spark-TTS实时流式合成失败: {e}")
            raise RuntimeError(f"TTS实时流式合成失败: {e}")

    def _generate_sentence_audio_optimized(self, text: str, speed: float, reference_audio: str) -> Optional[np.ndarray]:
        """优化的句子音频生成 - 针对实时性优化"""
        try:
            # 创建临时输出目录
            output_dir = Path("temp_audio")
            output_dir.mkdir(exist_ok=True)

            # 配置Spark-TTS参数 - 针对实时性优化，使用温柔女声设置
            voice = self.voice_settings["voice"]  # 温柔女声
            model_path = self.model_path

            # 生成唯一的文件前缀
            import time
            import threading
            thread_id = threading.get_ident()
            file_prefix = f"temp_audio/spark_tts_{thread_id}_{int(time.time() * 1000000)}"

            # 使用与直接调用完全相同的参数逻辑
            final_speed = speed  # 直接使用传入的速度，不做任何修改
            logger.info(f"🎵 实时方法使用直接调用逻辑: 传入速度={speed}, 最终速度={final_speed}")

            # 使用与直接调用完全相同的参数
            self.generate_audio(
                text=text,
                model_path=model_path,
                voice=None,  # 与直接调用保持一致
                speed=final_speed,  # 直接使用传入的速度
                lang_code='a',
                ref_audio=reference_audio if os.path.exists(reference_audio or "") else None,
                ref_text=self.reference_text,
                file_prefix=file_prefix,
                audio_format="wav",
                join_audio=True,
                play=False,
                verbose=False
            )

            # 读取生成的音频文件
            audio_file = Path(f"{file_prefix}.wav")
            if audio_file.exists():
                audio_data, _ = sf.read(str(audio_file))

                # 清理临时文件
                audio_file.unlink()

                # 确保音频数据格式正确
                if isinstance(audio_data, np.ndarray):
                    if audio_data.ndim == 2:  # 立体声转单声道
                        audio_data = np.mean(audio_data, axis=1)
                    return audio_data.astype(np.float32)

            return None

        except Exception as e:
            logger.error(f"❌ 优化句子音频生成失败: {e}")
            return None
