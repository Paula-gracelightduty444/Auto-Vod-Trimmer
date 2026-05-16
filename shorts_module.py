import os
import subprocess

def capture_thumbnail_burst(video_path, timestamp, event_id, spread, output_dir):
    """Pulls 3 frames from the exact mathematical center of the finalized Short."""
    os.makedirs(output_dir, exist_ok=True)
    offsets = [-spread, 0, spread] 
    for i, offset in enumerate(offsets):
        time_point = max(0, timestamp + offset)
        out_path = os.path.join(output_dir, f"Short_{event_id}_thumb_{i+1}.jpg")
        subprocess.run([
            'ffmpeg', '-ss', str(time_point), '-i', video_path,
            '-vframes', '1', '-q:v', '2', out_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def generate_shorts(ranked_clips, input_video, num_shorts, t_spread, thumb_dir, output_dir="Shorts"):
    if num_shorts <= 0 or not ranked_clips:
        return

    # Create the Shorts folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n [Shorts] Preparing to export up to {num_shorts} clips and thumbnails to '{output_dir}'...")

    exported_count = 0

    # Iterate through the entire VOD's ranked clips
    for idx, clip in enumerate(ranked_clips):
        # Stop completely if we have successfully created the desired number of shorts
        if exported_count >= num_shorts:
            break

        start_t = clip['start']
        end_t = clip['end']
        duration = end_t - start_t
        score = clip['score']

        # --- THE ANOMALY SHIELD: ENFORCE 120 - 180 SECOND RULE ---
        if duration < 120.0:
            print(f"   -> Skipping Clip Rank {idx+1}: {duration:.1f}s (Too short for combat criteria)")
            continue

        # If it passes the anomaly shield, enforce the 180-second hard ceiling
        if duration > 180.0:
            # Crop to 180 seconds, anchored to the center of the clip
            midpoint = start_t + (duration / 2.0)
            start_t = max(0.0, midpoint - 90.0)  
            end_t = start_t + 180.0              
            duration = 180.0                     

        # Use exported_count + 1 so the files are named 1 through X neatly
        out_file = os.path.join(output_dir, f"Short_{exported_count+1}_Score{score:.1f}.mp4")
        
        # FFmpeg command to quickly slice the video without heavy rendering
        # UPGRADED: Inherits the main engine's async audio resampling to prevent keyframe desync
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_t),
            '-to', str(end_t),
            '-i', input_video,
            '-c:v', 'copy',
            '-c:a', 'aac',                  # Explicitly re-encode the audio
            '-af', 'aresample=async=1',     # Force audio to sync to video keyframe timestamps
            out_file
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"   -> Created Short {exported_count+1}: {duration:.1f}s (Score: {score:.1f})")
        
        # --- NEW: TRIGGER THUMBNAIL BURST ON SUCCESSFUL SHORT ---
        short_center = start_t + (duration / 2.0)
        capture_thumbnail_burst(input_video, short_center, exported_count+1, t_spread, thumb_dir)
        
        # Successfully exported a valid clip and its thumbnails, increment the counter
        exported_count += 1
