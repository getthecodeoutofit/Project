import whisper
import moviepy.editor as mp
import cv2
import numpy as np
import tempfile
import os

def extract_audio(video_path):
    video = mp.VideoFileClip(video_path)
    temp_audio_path = tempfile.mktemp(suffix=".wav")
    video.audio.write_audiofile(temp_audio_path)
    return temp_audio_path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["segments"]

def add_captions_to_video(video_path, captions, font_scale=1.0):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output_path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    current_caption_index = 0
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        ret, frame = video.read()
        if not ret:
            break

        time_in_video = i / fps
        if current_caption_index < len(captions) and time_in_video >= captions[current_caption_index]["start"]:
            if time_in_video <= captions[current_caption_index]["end"]:
                text = captions[current_caption_index]["text"]
            else:
                current_caption_index += 1
                text = captions[current_caption_index]["text"] if current_caption_index < len(captions) else ""
            cv2.putText(
                frame, 
                text, 
                (10, frame_height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale,  # Use font_scale for caption size
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )

        out.write(frame)

    video.release()
    out.release()
    return temp_output_path

def combine_video_audio(video_path, audio_path, captioned_video_path):
    video = mp.VideoFileClip(captioned_video_path)
    audio = mp.AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_output_path = "final_captioned_video.mp4"
    final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
    return final_output_path

def add_captions(video_path, font_scale=1.0):
    audio_path = extract_audio(video_path)
    captions = transcribe_audio(audio_path)
    captioned_video_path = add_captions_to_video(video_path, captions, font_scale)
    final_video_path = combine_video_audio(video_path, audio_path, captioned_video_path)
    os.remove(audio_path)
    os.remove(captioned_video_path)
    return final_video_path

input_video_path = "C:/Project/project2/test.mp4"  # Update path to your video file
output_video_path = add_captions(input_video_path, font_scale=0.8)  # Adjust font_scale for caption size
print("Captioned video with audio saved at:", output_video_path)
