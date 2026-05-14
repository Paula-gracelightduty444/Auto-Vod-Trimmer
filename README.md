VOD Auto Trimmer
Overview

VOD Auto Trimmer is a multi-threaded Python engine designed to eliminate the manual labor of scrubbing through hours of stream footage. It uses localized AI speech-to-text and acoustic physics to autonomously analyze, rank, and edit raw VODs into ratio-controlled trimmed videos, highlight reels, and format-ready YouTube Shorts.

This tool runs 100% locally. It does not rely on cloud APIs, ensuring zero recurring costs and absolute privacy for your raw footage.
Core Capabilities

    Acoustic Intelligence (Librosa): Processes audio across 8 CPU cores to calculate dynamic range (Crest Factor), high-frequency combat impacts (Spectral Centroid > 1500Hz), and fundamental human pitch peaks (2000Hz+ for scream detection). It natively ignores static hiss and digital artifacts.

    Semantic Analysis (Faster-Whisper): Utilizes a 12-thread local AI model to transcribe microphone audio. It calculates Words-Per-Minute (WPM) to detect high-stress "Panic Modes," flags explicit hype words (e.g., "clip that"), and maps text back to 3-second acoustic chunks using word-level timestamps.

    The "Chaos Score" Algorithm: Ranks every 3-second segment of a VOD using a linear additive summation model. It weighs game volume, mic volume, combat triggers, linguistic intent, and WPM against a baseline floor to separate dead air from peak stream moments.

    Automated Output Generation:

        Trimmed VOD: Prunes the lowest-scoring "dead air" until the video matches a user-defined target ratio (e.g., 50% of the original duration).

        Highlight Reel: A tightly packed montage of the highest-scoring moments capped at a user-defined minute limit.

        Shorts Module: Extracts the absolute highest-ranking events, enforcing a 120-180 second rule (center-cropping clips that are too long) for vertical short-form platforms.

        Thumbnail Bursts: Automatically captures high-quality frames spread across the mathematical center of generated Shorts.

File Architecture

The project strictly requires the following naming conventions and file structure to execute successfully:

    Install_setup.bat: The primary deployment script. Installs required Python libraries and links FFmpeg via Winget.

    run_trimmer.bat: The execution script. Launches the GUI in an isolated Python process (pythonw) to prevent terminal closure interference.

    gui_master.py: A CustomTkinter interface handling all configuration, thresholds, and user inputs.

    vod_trimmer_master.py: The core mathematical engine that handles file parsing, multiprocessing, Whisper integration, smart-merging, and FFmpeg render commands.

    shorts_module.py: An autonomous sub-routine triggered by the Engine to handle the extraction, duration enforcement, and thumbnail generation of top-tier clips.

    config.ini (Auto-generated): Stores persistent threshold sliders and file paths.

Prerequisites

    OS: Windows 10/11

    Python: Python 3.12 installed. Crucial: You must check the box that says "Add Python to PATH" during installation.

Installation & Setup

    Download or clone this repository to a dedicated folder on your machine.

    Double-click Install_setup.bat.

        Note: This script will automatically utilize Windows Package Manager (Winget) to install FFmpeg and link your system environment variables. It will also use pip to install dependencies like numpy, librosa, and faster-whisper.

    Wait for the terminal to confirm "SETUP COMPLETE".

Usage

    Double-click run_trimmer.bat to open the Auto-VOD Editor GUI.

    Provide the absolute path to your source video (e.g., .mp4, .mkv).

    Define your OBS audio track indexes (e.g., Mic = 1, Game = 2). If dual tracks are unavailable, the engine will safely fallback to the master mix.

    Adjust your Thresholds (VOD Ratio, WPM panic gates, Scream Hz limits) or leave them at the mathematically tuned defaults.

    Click SAVE SETTINGS & RUN MASTER ENGINE.

    The terminal will open to provide heartbeat updates and diagnostic logs. Do not close this terminal until processing is complete.

All generated files (TrimmedVOD.mp4, highlightvid.mp4, and the Shorts/thumbnails folders) will output natively to the root directory where the script was executed.
Manual Bookmarking (The Yam Failsafe)

If an event occurs on stream that you want to guarantee makes it into the final render regardless of its acoustic or semantic score, speak your designated Bookmark Codeword (default: "Pineapple"). The AI semantic pass will flag this codeword, artificially inflate the segment's Chaos Score past the exception gate, and force its inclusion.