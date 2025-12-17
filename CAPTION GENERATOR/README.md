# Video Caption Generator

An automated video captioning tool that extracts audio from videos, transcribes it using OpenAI Whisper, optionally translates captions, and embeds them directly into the video with customizable styling.

## Features

- 🎤 **Automatic Speech Recognition**: Uses OpenAI Whisper for accurate transcription
- 🌍 **Language Detection**: Automatically detects the language of the audio
- 🔄 **Translation Support**: Optionally translate captions to a target language
- 🎨 **Customizable Styling**: Adjust font size, colors, outline, and position
- 📹 **Video Processing**: Processes video frames and embeds captions with proper timing
- 🔄 **Batch Processing**: Process multiple videos in a folder
- 📊 **Progress Tracking**: Visual progress bars for long-running operations

## Requirements

- Python 3.8 or higher
- FFmpeg (required by MoviePy)
- CUDA-capable GPU (optional, for faster Whisper processing)

## Installation

1. **Install FFmpeg** (if not already installed):
   - **Linux**: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or `sudo pacman -S ffmpeg` (Arch)
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)

2. **Clone or download this repository**

3. **Activate the conda environment**:
   ```bash
   conda activate caption
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

1. **Activate the conda environment**:
   ```bash
   conda activate caption
   ```

2. **Edit the `CHECK.py` file** and update the `input_video_path` variable, then run:
   ```bash
   python CHECK.py
   ```

### Programmatic Usage

```python
from CHECK import add_captions

# Add captions to a video
output_path = add_captions(
    video_path="path/to/your/video.mp4",
    font_scale=0.8,
    color=(255, 255, 255),  # White text (BGR format)
    outline_color=(0, 0, 0),  # Black outline (BGR format)
    position="bottom",  # "bottom" or "top"
    target_language="en"  # Target language code, or None to keep original
)
```

### Batch Processing

```python
from CHECK import batch_process_videos

# Process all .mp4 files in a folder
batch_process_videos(
    folder_path="path/to/videos/",
    font_scale=0.8,
    color=(255, 255, 255),
    outline_color=(0, 0, 0),
    position="bottom",
    target_language="en"
)
```

## Parameters

### `add_captions()` Function

- **`video_path`** (str): Path to the input video file
- **`font_scale`** (float, default=1.0): Font size multiplier. Larger values = bigger text
- **`color`** (tuple, default=(255, 255, 255)): Text color in BGR format (Blue, Green, Red)
- **`outline_color`** (tuple, default=(0, 0, 0)): Text outline color in BGR format
- **`position`** (str, default="bottom"): Caption position - "bottom" or "top"
- **`target_language`** (str, optional): Target language code (e.g., "en", "es", "fr"). If None, keeps the detected language

### Language Codes

Common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ja` - Japanese
- `zh` - Chinese
- `ko` - Korean
- `ar` - Arabic

See [Google Translate language codes](https://cloud.google.com/translate/docs/languages) for a complete list.

## How It Works

1. **Audio Extraction**: Extracts audio from the video file
2. **Transcription**: Uses Whisper to transcribe the audio with timestamps
3. **Language Detection**: Detects the language of the transcription
4. **Translation** (optional): Translates captions if a target language is specified
5. **Caption Rendering**: Processes each video frame and adds captions at the appropriate timestamps
6. **Audio Recombination**: Combines the captioned video with the original audio
7. **Cleanup**: Removes temporary files

## Features Details

### Dynamic Font Scaling
The script automatically adjusts font size based on caption length:
- Text > 100 characters: Font scale = 0.5
- Text > 50 characters: Font scale = 0.6
- Otherwise: Uses the specified `font_scale` parameter

### Error Handling
- Handles missing or empty transcriptions
- Gracefully handles translation failures (keeps original text)
- Proper resource cleanup even if errors occur

## Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   - Ensure FFmpeg is installed and in your system PATH
   - Verify installation: `ffmpeg -version`

2. **CUDA/GPU issues**:
   - Whisper will use CPU if CUDA is not available (slower but works)
   - Install CUDA-enabled PyTorch for GPU acceleration

3. **Memory errors**:
   - Large videos may require significant RAM
   - Consider processing shorter segments or using a smaller Whisper model

4. **Translation errors**:
   - Check your internet connection (Google Translate API requires internet)
   - Some language pairs may not be supported

## Performance Notes

- Processing time depends on video length and hardware
- GPU acceleration significantly speeds up Whisper transcription
- Larger Whisper models (base, small, medium, large) provide better accuracy but are slower
- Current default model: "base" (good balance of speed and accuracy)

## License

This project is open source. Feel free to modify and use as needed.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [MoviePy](https://github.com/Zulko/moviepy) for video processing
- [OpenCV](https://opencv.org/) for video frame processing
- [Deep Translator](https://github.com/nidhaloff/deep-translator) for translation services

