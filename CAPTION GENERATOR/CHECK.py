import whisper
try:
    # MoviePy 2.x
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
except ImportError:
    # MoviePy 1.x fallback
    import moviepy.editor as mp
    VideoFileClip = mp.VideoFileClip
    AudioFileClip = mp.AudioFileClip
import cv2
import numpy as np
import tempfile
import os
import logging
from tqdm import tqdm
from deep_translator import GoogleTranslator
from langdetect import detect
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_audio(video_path):
    """Extracts audio from video and saves it as a temporary .wav file."""
    logging.info("Extracting audio from video...")
    try:
        video = VideoFileClip(video_path)
        temp_audio_path = tempfile.mktemp(suffix=".wav")
        video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
        video.close()
        logging.info("Audio extracted successfully.")
        return temp_audio_path
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        raise

def detect_language(segments):
    """Detects the language of the audio transcription."""
    if not segments or len(segments) == 0:
        logging.warning("No segments found, defaulting to 'en'")
        return "en"
    first_segment_text = segments[0]["text"]
    if not first_segment_text or len(first_segment_text.strip()) == 0:
        logging.warning("Empty segment text, defaulting to 'en'")
        return "en"
    try:
        detected_lang = detect(first_segment_text)
        logging.info(f"Detected language: {detected_lang}")
        return detected_lang
    except Exception as e:
        logging.warning(f"Language detection failed: {e}, defaulting to 'en'")
        return "en"

def transcribe_and_translate_audio(audio_path, target_language=None):
    """Transcribes audio, optionally translating it into the target language."""
    logging.info("Transcribing audio...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    segments = result["segments"]
    logging.info(f"Transcription completed with {len(segments)} segments.")
    
    detected_language = detect_language(segments)
    if target_language and detected_language != target_language:
        logging.info("Translating captions to target language...")
        for segment in segments:
            try:
                segment["text"] = GoogleTranslator(source=detected_language, target=target_language).translate(segment["text"])
            except Exception as e:
                logging.warning(f"Translation failed for segment '{segment['text']}': {e}")
                segment["text"] = segment["text"]  # Keep original text if translation fails

    return segments

def add_captions_to_video(video_path, captions, font_scale=1.0, color=(255, 255, 255), outline_color=(0, 0, 0), position="bottom"):
    """Adds captions to a video based on the provided transcription segments."""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output_path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    current_caption_index = 0
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    position_offset = 30  # Offset from bottom or top edge

    logging.info("Adding captions to video...")
    for i in tqdm(range(num_frames), desc="Processing frames"):
        ret, frame = video.read()
        if not ret:
            break

        time_in_video = i / fps
        
        # Update current caption index if we've passed the end of the current caption
        while current_caption_index < len(captions) and time_in_video > captions[current_caption_index]["end"]:
            current_caption_index += 1
        
        # Determine if we should display a caption
        text = ""
        if current_caption_index < len(captions):
            caption = captions[current_caption_index]
            if caption["start"] <= time_in_video <= caption["end"]:
                text = caption["text"]

        # Dynamic font scaling based on text length
        current_font_scale = font_scale
        if text:
            if len(text) > 100:
                current_font_scale = 0.5
            elif len(text) > 50:
                current_font_scale = 0.6

            # Position adjustment for captions to avoid covering faces or important visuals
            text_x = 10  # X coordinate for the text
            if position == "bottom":
                text_y = frame_height - position_offset
            else:
                text_y = position_offset

            # Add text with outline for better visibility
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, current_font_scale, 2)
            # Outline
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, current_font_scale, outline_color, 6, cv2.LINE_AA)
            # Main text
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, current_font_scale, color, 2, cv2.LINE_AA)

        out.write(frame)

    video.release()
    out.release()
    logging.info("Captions added to video.")
    return temp_output_path

def combine_video_audio(video_path, audio_path, captioned_video_path):
    """Combines the video with captions and the original audio."""
    logging.info("Combining video with audio...")
    video = None
    audio = None
    final_video = None
    try:
        video = VideoFileClip(captioned_video_path)
        audio = AudioFileClip(audio_path)
        # MoviePy 2.x uses with_audio instead of set_audio
        try:
            final_video = video.with_audio(audio)
        except AttributeError:
            # Fallback for MoviePy 1.x
            final_video = video.set_audio(audio)
        final_output_path = tempfile.mktemp(suffix=".mp4")
        final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
        logging.info("Video and audio combined successfully.")
        return final_output_path
    except Exception as e:
        logging.error(f"Error combining video and audio: {e}")
        raise
    finally:
        # Clean up resources
        if final_video is not None:
            final_video.close()
        if audio is not None:
            audio.close()
        if video is not None:
            video.close()

def add_captions(video_path, font_scale=1.0, color=(255, 255, 255), outline_color=(0, 0, 0), position="bottom", target_language=None):
    """Main function to add captions to the video, handling transcription, translation, and rendering."""
    audio_path = extract_audio(video_path)
    captions = transcribe_and_translate_audio(audio_path, target_language=target_language)
    captioned_video_path = add_captions_to_video(video_path, captions, font_scale, color, outline_color, position)
    final_video_path = combine_video_audio(video_path, audio_path, captioned_video_path)
    
    # Clean up temporary files
    os.remove(audio_path)
    os.remove(captioned_video_path)
    logging.info(f"Final captioned video saved at {final_video_path}")
    return final_video_path

def batch_process_videos(folder_path, font_scale=1.0, color=(255, 255, 255), outline_color=(0, 0, 0), position="bottom", target_language=None):
    """Batch process videos in a folder, adding captions to each video."""
    logging.info("Starting batch processing of videos...")
    for video_file in os.listdir(folder_path):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(folder_path, video_file)
            logging.info(f"Processing video: {video_file}")
            try:
                add_captions(video_path, font_scale, color, outline_color, position, target_language)
            except Exception as e:
                logging.error(f"Failed to process video {video_file}: {e}")

# Example usage
if __name__ == "__main__":
    input_video_path = "test.mp4"  # Update path to your video file
    if os.path.exists(input_video_path):
        output_video_path = add_captions(
            input_video_path,
            font_scale=0.8,
            color=(255, 255, 255),
            outline_color=(0, 0, 0),
            position="bottom",
            target_language="en"  # Set to 'es' for Spanish translation, or None to keep the detected language
        )
        print("Captioned video with audio saved at:", output_video_path)
    else:
        print(f"Error: Video file '{input_video_path}' not found. Please update the path.")
