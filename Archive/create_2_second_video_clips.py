import os
import cv2

def split_video_into_clips(video_path, output_dir, clip_length_seconds=2, required_fps=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(original_fps - required_fps) > 1:
        print(f"[SKIP] {os.path.basename(video_path)} has FPS={original_fps:.2f}, not ~{required_fps}")
        cap.release()
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_length_frames = clip_length_seconds * required_fps

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Processing {os.path.basename(video_path)} ({total_frames} frames, {width}x{height}, {original_fps:.2f} FPS)")

    clip_count = 0
    while True:
        frames = []
        for _ in range(clip_length_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if len(frames) < clip_length_frames:
            break  # End of video

        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_clip_{clip_count:04d}.avi")
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), required_fps, (width, height))
        for f in frames:
            out.write(f)
        out.release()

        print(f"[SAVED] {output_file}")
        clip_count += 1

    cap.release()
    print(f"[DONE] {os.path.basename(video_path)} split into {clip_count} clips\n")

def batch_process_ucf101(input_dir, output_dir_base, clip_length_seconds=2, required_fps=30):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".avi"):
                video_path = os.path.join(root, file)
                # Maintain subdirectory structure
                relative_path = os.path.relpath(root, input_dir)
                output_dir = os.path.join(output_dir_base, relative_path)
                split_video_into_clips(video_path, output_dir, clip_length_seconds, required_fps)

if __name__ == "__main__":
    # Update paths below
    input_dir = "./UCF101"                 # Root directory containing all UCF101 videos
    output_dir_base = "./UCF101_clips"     # Where the 2s clips will be saved

    batch_process_ucf101(input_dir, output_dir_base)
