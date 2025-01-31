"""
v0.1.0 2025-01-28
重构说明：
1. 清理未使用的导入和重复声明
2. 修复变量作用域问题
3. 增强异常处理和防御性编程
4. 优化代码结构和PEP8合规性
5. 添加线程安全机制
6. 完善文件清理和资源管理
"""

import os
import logging
import re
import tempfile
import filetype
import ffmpeg
import torch
import gradio as gr

from typing import List
from faster_whisper import WhisperModel
from threading import Lock
from typing import List, Generator, Any
import socket
from contextlib import closing

# 配置环境变量防止CUDA库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 媒体处理配置 (常量使用全大写，如果出现最大文件问题可以调整MAX_FILE_SIZE参数
MEDIA_CONFIG = {
    "MAX_FILE_SIZE": 1024 * 1024 * 8192,  # 8192MB
    "ALLOWED_AUDIO": ["mp3", "wav", "aac", "flac", "ogg", "m4a"],
    "ALLOWED_VIDEO": ["mp4", "avi", "mov", "mkv", "webm", "flv"],
    "TEMP_DIR": "temp_media",
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 0.5
}

# 模型初始化锁防止竞态条件
MODEL_INIT_LOCK = Lock()

# 初始化全局状态
class ProcessingState:
    def __init__(self):
        self.is_cancelled = False
        self.active_threads = 0

# 创建全局状态实例
processing_state = ProcessingState()

def initialize_model():
    """线程安全的模型初始化"""
    with MODEL_INIT_LOCK:
        model_size = "large-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        return WhisperModel(model_size, device=device, compute_type=compute_type)

# 延迟加载模型
_model_instance = None

def get_model():
    """单例模式获取模型实例"""
    global _model_instance
    if _model_instance is None:
        _model_instance = initialize_model()
    return _model_instance

def validate_media_file(file_path: str) -> tuple:
    """验证媒体文件格式和大小"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size > MEDIA_CONFIG["MAX_FILE_SIZE"]:
        raise ValueError(
            f"文件大小超过限制: {MEDIA_CONFIG['MAX_FILE_SIZE']//(1024*1024)}MB"
        )

    kind = filetype.guess(file_path)
    if not kind:
        raise ValueError("无法识别的文件类型")

    if kind.extension in MEDIA_CONFIG["ALLOWED_AUDIO"]:
        return "audio", kind.extension
    if kind.extension in MEDIA_CONFIG["ALLOWED_VIDEO"]:
        return "video", kind.extension

    raise ValueError(f"不支持的文件格式: {kind.extension}")

def extract_audio_from_video(video_path: str) -> str:
    """使用ffmpeg从视频中提取音频"""
    try:
        os.makedirs(MEDIA_CONFIG["TEMP_DIR"], exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(MEDIA_CONFIG["TEMP_DIR"], f"{base_name}.wav")

        (
            ffmpeg.input(video_path)
            .output(output_path, ac=1, ar=16000)
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        return output_path
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg错误: {e.stderr.decode()}")
        raise RuntimeError("音频提取失败，请检查文件格式是否支持") from e

class LogHandler(logging.Handler):
    """自定义日志处理器，支持关键字高亮"""
    HIGHLIGHT_PATTERNS = [
        (r'\b(error|fail)\b', 'error'),
        (r'\b(warning)\b', 'warning'),
        (r'\b(info)\b', 'info'),
        (r'\b(debug)\b', 'debug')
    ]

    def __init__(self):
        super().__init__()
        self.logs = []
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        try:
            log_entry = self.format(record)
            # 关键字高亮处理
            for pattern, _ in self.HIGHLIGHT_PATTERNS:
                if re.search(pattern, log_entry, re.IGNORECASE):
                    log_entry = f"**{log_entry}**"
                    break
            self.logs.append(log_entry)
            print(log_entry)  # 同时输出到控制台
        except Exception as e:
            print(f"日志记录失败: {str(e)}")

# 初始化日志系统
log_handler = LogHandler()
logging.basicConfig(
    level=logging.INFO,
    handlers=[log_handler],
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def safe_process_file(audio_path: str) -> str:
    """安全处理音频文件，带有资源清理和取消支持"""
    tmp_path = None
    model = get_model()  # 获取初始化好的模型

    try:
        # 创建本地文件副本防止原始文件被修改
        # 使用安全方式创建临时文件（Windows兼容）
        with open(audio_path, "rb") as src_file:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                dir=MEDIA_CONFIG["TEMP_DIR"]
            ) as tmp_file:
                tmp_file.write(src_file.read())
                tmp_path = tmp_file.name
        os.chmod(tmp_path, 0o666)  # 设置文件权限

        logging.info(f"开始处理文件: {os.path.basename(tmp_path)}")

        if processing_state.is_cancelled:
            logging.warning("处理已被用户取消")
            return "处理已取消"

        # 执行语音识别
        segments, info = model.transcribe(tmp_path)
        logging.info(f"检测到语言: {info.language} (置信度: {info.language_probability:.2%})")

        transcription = []
        for segment in segments:
            if processing_state.is_cancelled:
                logging.warning("处理已被用户取消")
                return "处理已取消"

            logging.debug(f"分段: {segment.start:.2f}-{segment.end:.2f} - {segment.text}")
            transcription.append(segment.text.strip())

        full_transcription = " ".join(transcription)
        
        # 自动保存结果文件
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取原始文件名（保留输入文件扩展名前的主体名称）
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 生成双格式文件
        for ext in ['.txt', '.md']:
            output_path = os.path.join(output_dir, f"{original_name}{ext}")
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    if ext == '.md':
                        f.write(f"# 语音转录结果\n**文件**: `{original_name}`\n\n")
                        f.write(full_transcription)
                    else:
                        f.write(full_transcription)
                logging.info(f"结果已保存至：{output_path}")
            except IOError as e:
                logging.error(f"文件保存失败 {ext}: {str(e)}")
        
        return full_transcription
    except Exception as e:
        
        logging.error(f"处理失败: {str(e)}")
        return f"处理错误: {str(e)}"
    finally:
        # 确保清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as cleanup_error:
                logging.warning(f"临时文件清理失败: {str(cleanup_error)}")

# 类型注解修正后的batch_transcribe函数
def batch_transcribe(files: List[gr.components.File], batch_size: int) -> Generator[str, Any, None]:
    """批量处理音频文件，支持取消操作"""
    processing_state.is_cancelled = False
    results = []
    
    try:
        for i, file in enumerate(files, 1):
            if processing_state.is_cancelled:
                break

            logging.info(f"正在处理文件 {i}/{len(files)}: {file.name}")
            try:
                result = safe_process_file(file.name)
                results.append(f"文件 {i}: {result}")
            except Exception as e:
                results.append(f"文件 {i} 错误: {str(e)}")

            if i % batch_size == 0:
                yield "\n".join(results)
                results = []
                
        if results:
            yield "\n".join(results)
            
    finally:
        processing_state.is_cancelled = False



# Gradio界面配置
def create_interface():
    """创建包含完整事件绑定的Gradio界面"""
    with gr.Blocks(title="语音转录系统") as interface:
        gr.Markdown("# 高效语音转录系统 (基于Faster-Whisper)")

        with gr.Row():
            # 单文件处理面板
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="上传单个文件")
                transcribe_btn = gr.Button("开始转录", variant="primary")
                stop_btn = gr.Button("停止处理", variant="stop")
                gr.Markdown("### 转录结果")
                transcription_output = gr.Textbox(label="", lines=20)

            # 批量处理面板
            with gr.Column():
                file_input = gr.File(label="批量上传文件", file_count="multiple",height=190)
                batch_size = gr.Slider(1, 50, value=1, step=1, label="单次处理批量大小")
                start_btn = gr.Button("开始批量处理", variant="primary")
                stop_btn = gr.Button("停止批量处理", variant="stop")
                gr.Markdown("### 批量处理结果")
                output_area = gr.Textbox(label="", lines=20)

        # 日志面板
        with gr.Row():
            log_filter = gr.Dropdown(
                ["All", "Info", "Warning", "Error", "Debug"],
                value="All",
                label="日志过滤"
            )
            refresh_btn = gr.Button("刷新日志")
 # 新增日志显示组件开始
        with gr.Row():
            log_output = gr.Textbox(
                label="日志（若未自动输出请点击刷新）",
                lines=15,
                interactive=True,
                autoscroll=True,
                elem_id="live_logs"
            )
        
        # 设置日志更新回调
        log_handler.update_event = lambda: log_output.update(
            value="\n".join(log_handler.logs[-100:]),
            every=0.5
        )
        # 事件绑定（在同一个作用域内完成）
        transcribe_btn.click(
            safe_process_file,
            inputs=audio_input,
            outputs=transcription_output
        )

        start_btn.click(  # 现在start_btn在作用域内可见
            batch_transcribe,
            inputs=[file_input, batch_size],
            outputs=output_area,
            show_progress="full"
        )

        stop_btn.click(
            lambda: setattr(processing_state, "is_cancelled", True),
            outputs=None
        )

        refresh_btn.click(
            lambda level: "\n".join(log_handler.logs if level == "All" else
                [log for log in log_handler.logs if f"- {level.upper()} -" in log]),
            inputs=log_filter,
            outputs=log_output
        )

    return interface

def find_free_port(start_port: int = 7860, end_port: int = 8000) -> int:
    """自动寻找可用端口"""
    for port in range(start_port, end_port + 1):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise OSError(f"No available ports between {start_port}-{end_port}")

if __name__ == "__main__":
    # 确保临时目录存在
    os.makedirs(MEDIA_CONFIG["TEMP_DIR"], exist_ok=True)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)  # 插入位置

    # 自动选择端口
    selected_port = find_free_port(7860, 7870)
    
    try:
        interface = create_interface()
        print(f"🌐 启动服务: http://localhost:{selected_port}")
        interface.launch(
            server_port=selected_port,
            share=False,
            show_error=True,
            server_name="localhost",
            prevent_thread_lock=True
        )
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        print("尝试备用端口...")
        selected_port = find_free_port(7871, 7890)
        interface.launch(
            server_port=selected_port,
            share=False,
            show_error=True,
            server_name="localhost"
        )
    finally:
        print(f"✅ 服务运行在端口: {selected_port}")
        interface.close()  # 确保释放资源