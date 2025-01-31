# Faster-Whisper WebUI

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![OS Support](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

## 项目概述
基于Faster-Whisper的语音转录解决方案，提供以下功能：

**核心功能**
- 支持音频/视频文件转录 (MP3, WAV, MP4等格式)
- 自动语音识别与文本生成
- 批量文件处理功能
- 实时日志监控与错误处理

**技术栈**
| 组件 | 用途 |
|------|------|
| Python 3.10+ | 后端逻辑 |
| Gradio | Web交互界面 |
| Faster-Whisper | 语音识别引擎 |
| FFmpeg | 音视频处理 |

## 环境要求

### 硬件配置
- NVIDIA GPU (推荐) 或 CPU
- 最小4GB显存 (GPU模式)
- 16GB内存

### 软件依赖
- [Python 3.10+](https://www.python.org/)
- [FFmpeg](https://ffmpeg.org/)

## 安装指南

```bash
# 克隆仓库
git clone https://github.com/AdamPlatin123/Faster-Whisper-WebUI.git
cd Faster-Whisper-WebUI

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 设置环境变量(Windows)
set KMP_DUPLICATE_LIB_OK=TRUE

# 运行应用
python app.py
```

## 配置管理

### 环境变量
| 变量名 | 描述 | 默认值 | 必填 |
|--------|------|--------|------|
| KMP_DUPLICATE_LIB_OK | 解决CUDA库冲突 | TRUE | 是 |

### 文件存储配置
```python
MEDIA_CONFIG = {
    "MAX_FILE_SIZE": 8192,  # MB
    "ALLOWED_AUDIO": ["mp3","wav","aac","flac","ogg","m4a"],
    "TEMP_DIR": "temp_media"  # 临时文件目录
}
```

## 使用说明

### 启动选项
```bash
# 常规模式
python app.py

# 调试模式(显示详细日志)
python app.py --debug
```

### 访问界面
应用启动后访问：
```
http://localhost:7860 (程序会自动选择可用端口)
```

## 依赖管理

### 主要依赖
| 包名 | 版本 | 用途 |
|------|------|------|
| gradio | >=3.0.0 | Web界面框架 |
| faster-whisper | >=0.9.0 | 语音识别核心 |
| torch | >=2.0.0 | GPU加速支持 |

完整依赖见[requirements.txt](./requirements.txt)

## 常见问题

### 错误处理
| 错误现象 | 解决方案 |
|---------|----------|
| CUDA初始化失败 | 1. 确认NVIDIA驱动正常<br>2. 执行`set KMP_DUPLICATE_LIB_OK=TRUE` |
| 文件格式不支持 | 检查文件扩展名是否在ALLOWED_AUDIO/VIDEO列表内 |
| 内存不足 | 减小MAX_FILE_SIZE配置值 |

### 依赖冲突解决
```bash
# 创建干净虚拟环境
python -m venv clean_venv
clean_venv\Scripts\activate
pip install -r requirements.txt
### 启动选项
# 常规模式
python app.py
# 调试模式(显示详细日志)
python app.py --debug
```

## 许可协议
本项目遵循 [Apache 2.0 License](LICENSE)