# Faster-Whisper WebUI

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![OS Support](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

## Project Overview
A speech transcription solution based on Faster-Whisper with the following features:

**Core Features**
- Support audio/video file transcription (MP3, WAV, MP4 formats)
- Automatic speech recognition and text generation (Markdown & TXT formats)
- Batch file processing
- Real-time log monitoring and error handling

**Technology Stack**
| Component         | Purpose                |
|-------------------|------------------------|
| Python 3.10+      | Backend logic          |
| Gradio            | Web interface          |
| Faster-Whisper    | Speech recognition engine |
| FFmpeg            | Audio/video processing |

## Requirements

### Hardware
- NVIDIA GPU (recommended) or CPU
- Minimum 4GB VRAM (GPU mode)
- 16GB RAM

### Software Dependencies
- [Python 3.10+](https://www.python.org/)
- [FFmpeg](https://ffmpeg.org/)

## Installation

```bash
# Clone repository
git clone https://github.com/AdamPlatin123/Faster-Whisper-WebUI.git
cd Faster-Whisper-WebUI

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (Windows)
set KMP_DUPLICATE_LIB_OK=TRUE

# Run application
python app.py
```

## Configuration

### Environment Variables
| Variable Name            | Description                     | Default | Required |
|--------------------------|---------------------------------|---------|----------|
| KMP_DUPLICATE_LIB_OK     | Resolve CUDA library conflicts  | TRUE    | Yes      |

### File Storage
```python
MEDIA_CONFIG = {
    "MAX_FILE_SIZE": 8192,  # MB
    "ALLOWED_AUDIO": ["mp3","wav","aac","flac","ogg","m4a"],
    "TEMP_DIR": "temp_media"  # Temporary directory
}
```

## Usage

### Startup Options
```bash
# Normal mode
python app.py

# Debug mode (verbose logging)
python app.py --debug
```

### Access Interface
After startup, access via:
```
http://localhost:7860 (auto-selects available port)
```

## Dependencies

### Main Packages
| Package         | Version   | Purpose                |
|-----------------|-----------|------------------------|
| gradio          | >=3.0.0   | Web interface framework|
| faster-whisper  | >=0.9.0   | Core speech recognition|
| torch           | >=2.0.0   | GPU acceleration       |

Full dependencies see [requirements.txt](./requirements.txt)

## FAQ

### Troubleshooting
| Error Symptom            | Solution                          |
|--------------------------|-----------------------------------|
| CUDA initialization fail | 1. Verify NVIDIA drivers<br>2. Execute `set KMP_DUPLICATE_LIB_OK=TRUE` |
| Unsupported file format  | Check file extension against ALLOWED_AUDIO list |
| Insufficient memory      | Reduce MAX_FILE_SIZE value        |

### Resolving Dependency Conflicts
```bash
# Create clean environment
python -m venv clean_venv
clean_venv\Scripts\activate
pip install -r requirements.txt
```

## License
This project is licensed under the [Apache 2.0 License](LICENSE)