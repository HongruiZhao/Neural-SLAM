import cv2
import os
import argparse
from pathlib import Path
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def make_video(image_folder, video_name, frame_rate):
    """
    Create a video from a folder of images.
    """
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    if not images:
        print("No images found in the specified folder.")
        return

    images = sorted_alphanumeric(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    print(f"Video '{video_name}' created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from a folder of images.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("--video_name", type=str, default="video.mp4", help="Name of the output video file.")
    parser.add_argument("--frame_rate", type=int, default=10, help="Frame rate of the video.")
    args = parser.parse_args()

    make_video(args.image_folder, args.video_name, args.frame_rate)
