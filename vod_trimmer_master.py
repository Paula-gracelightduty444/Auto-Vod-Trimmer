import os
import sys
import subprocess
import numpy as np
import soundfile as sf
import librosa
import threading
import time
import configparser
import io
import re
import logging
import concurrent.futures
import shorts_module
import socket
import random

from tqdm import tqdm

# ==========================================
# 1. SYSTEM & CONFIGURATION INITIALIZATION
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING) # Silences the HuggingFace HTTP GET Request spam

if getattr(sys, 'frozen', False):
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.ini")
DEFAULT_CONFIG = {
    'PATHS': {
        'input_video': 'my_stream.mp4',
        'highlights_output': 'highlightvid.mp4',
        'montage_output': 'TrimmedVOD.mp4',
        'thumbnail_folder': 'thumbnails'
    },
    'OBS_TRACKS': {
        'mic_track': '1',
        'game_track': '2'
    },
    'THRESHOLDS': {
        'highlight_max_minutes': '55',
        'highlight_exception_score': '100',
        'vod_normal_ratio': '0.75',
        'vod_keep_base_score': '7',
        'panic_wpm_threshold': '200',
        'scream_threshold_hz': '2000',
        'scream_min_seconds': '1.0',
        'thumbnail_spread': '6.0',
        'thumbnail_events': '5',
        'clip_buffer_seconds': '2.0',
        'max_clip_length': '180.0',
        'max_merge_gap': '12.0'
    },
    'BOOKMARK': {
        'codeword': 'Pineapple'
    },
    'VOCAB': {
        'hype_words': 'clip that, oh fuck, oh shit, reloading, grenade'
    },
    'SHORTS': {
        'num_shorts': '10'
    }
}

_instance_lock = None

def load_or_create_config():
    config = configparser.ConfigParser()
    config.read_dict(DEFAULT_CONFIG)  # load defaults first
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f:
            config.write(f)
    config.read(CONFIG_FILE)  # user values override defaults
    return config

def acquire_instance_lock():
    global _instance_lock
    _instance_lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _instance_lock.bind(('127.0.0.1', 65432))
    except socket.error:
        print("\n" + "!"*70)
        print(" [CRITICAL ERROR] VOD Trimmer is already running in another window!")
        print("!"*70)
        print("\nRunning multiple instances will overload your CPU and crash your PC.")
        print("Please close the other terminal window before starting a new run.")
        input("\nPress Enter to exit safely...")
        sys.exit(1)

# ==========================================
# 2. VISUAL FEEDBACK (HEARTBEAT)
# ==========================================
class Heartbeat:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None

    def _pulse(self, message):
        while not self.stop_event.is_set():
            for dots in [".  ", ".. ", "...", ".. "]:
                if self.stop_event.is_set(): break
                sys.stdout.write(f"\r     -> {message}{dots}")
                sys.stdout.flush()
                time.sleep(0.4)

    def start(self, message):
        if self.thread and self.thread.is_alive(): return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._pulse, args=(message,))
        self.thread.start()

    def stop(self):
        if not self.thread or not self.thread.is_alive(): return
        self.stop_event.set()
        self.thread.join()
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

beat = Heartbeat()

# ==========================================
# 3. MULTIPROCESSING & MATH ALGORITHMS
# ==========================================
def calculate_prosody(pitch_values_speech):
    # pitch_values_speech: pre-filtered to 80-400Hz speech range, avoids a second piptrack call
    if len(pitch_values_speech) < 20: return 1.0
    std = np.std(pitch_values_speech)
    # Raised monotone penalty threshold to std < 30 to match human vocal biology
    return 1.5 if std > 50 else (0.7 if std < 30 else 1.0)

def calculate_chaos_score(v_game, g_centroid, g_onset, g_crest, v_mic, mic_pitch_peak, is_scream,
                          is_panic, prosody, transcription, hype_patterns, laugh_pattern, bookmark_pattern, exception_gate,
                          speech_ratio, seg_duration=3.0):

    # --- Linguistic Processing & WPM Diagnostic ---
    found_text = transcription.lower()
    word_count = len(found_text.split())
    average_wpm = (word_count / seg_duration) * 60 if seg_duration > 0 else 0

    if average_wpm > 600.0:
         logging.warning(f"[WPM DIAGNOSTIC] Potential AI Hallucination Loop! WPM: {average_wpm:.1f}. Text: '{found_text[:50]}...'")

    # --- Volume Bonuses ---
    vocal_vol_bonus = v_mic * 2.5
    game_vol_bonus = v_game * 2.5

    # --- RANGE BEAM 1: Scream Detection (PITCH) ---
    scream_bonus = 8.0 if (is_scream or (2000 < mic_pitch_peak < 4000)) else 0.0

    # --- THE WALL OF NOISE SHIELD ---
    # Lowered back to 1.3 to still filter out low volume hums but retain supressed gunfire.
    # Gunfights and explosive impacts have sharp volume spikes (Crest factor > 2.5+).
    is_wall_of_noise = (g_crest < 1.3)

    if is_wall_of_noise:
        combat_bonus = 0.0
        impact_bonus = 0.0
        gore_bonus = 0.0
    else:
        # --- RANGE BEAM 2: Combat Detection (CENTROID) ---
        # BUFFED: Bonuses increased to 6.0 to prioritize game action over chatter
        combat_bonus = 6.0 if (1500 < g_centroid < 6000) else 0.0
        impact_bonus = 6.0 if g_onset > 5.0 else 0.0

        # --- RANGE BEAM 2.5: Melee / Gore Detection ---
        # BUFFED: Gore weight increased to 6.0 to match combat intensity
        gore_bonus = 6.0 if (500 < g_centroid <= 1500) and (g_onset > 4.0) else 0.0

    # --- NEW: TRUE COMBAT MULTIPLIER ---
    is_heavy_combat = (combat_bonus > 0 and impact_bonus > 0) or (gore_bonus > 0)
    combat_multiplier = 1.7 if is_heavy_combat else 1.0

    prosody_bonus = 4.0 if prosody > 1.0 else (-2.0 if prosody < 1.0 else 0.0)
    speech_bonus = speech_ratio * 2.5

    # --- NLP INTENT: HYPE WORDS ---
    n_matches = sum(1 for p in hype_patterns if p.search(found_text))

    # --- RANGE BEAM 3: Laughter Detection ---
    is_laughing = bool(laugh_pattern.search(found_text))
    laugh_bonus = 2.0 if is_laughing else 0.0

    # Base floor of 2.0
    sub_total = 2.0 + vocal_vol_bonus + game_vol_bonus + scream_bonus + combat_bonus + gore_bonus + prosody_bonus + speech_bonus + impact_bonus + laugh_bonus

    # --- NEW: HYPE MULTIPLIER (Human Intent Flag) ---
    # If the user explicitly uses a combat callout, stretch the final score by 1.5x
    hype_multiplier = 1.5 if n_matches > 0 else 1.0

    # --- TRIPLE MULTIPLIER STACK ---
    panic_multiplier = 1.3 if is_panic else 1.0
    final_score = sub_total * panic_multiplier * combat_multiplier * hype_multiplier

    is_bookmarked = bool(bookmark_pattern.search(found_text))
    if is_bookmarked:
        final_score = max(final_score, exception_gate + 2.0)

    return final_score, n_matches, is_bookmarked

def analyze_acoustic_chunk(args):
    i, c_mic, c_game, sr, seg_len, s_thresh, s_min = args
    
    # --- NEW: CREST FACTOR (DYNAMIC RANGE) ---
    game_rms_array = librosa.feature.rms(y=c_game)[0]
    v_game = np.mean(game_rms_array)
    # Crest Factor = Peak Volume / Average Volume
    g_crest = np.max(game_rms_array) / (v_game + 0.0001) 
    
    v_mic = np.mean(librosa.feature.rms(y=c_mic))
    
    # Notice we added two 0.0s to the end of the return to represent empty pitch data
    if v_mic < 0.01 and v_game < 0.01:
        # Returning 1.0 for default crest factor in empty chunks
        return (i, v_game, v_mic, 0.0, 1.0, 0.0, 1.0, False, 0.0, 0.0)
        
    # --- HARMONIC-PERCUSSIVE SEPARATION (HPSS) ---
    # Split the game audio. We discard the harmonic (music/drone) and keep the percussive (impacts/combat)
    _, c_game_percussive = librosa.effects.hpss(c_game)

    # Use the separated percussive track for Centroid and Onset calculations
    g_c = np.mean(librosa.feature.spectral_centroid(y=c_game_percussive, sr=sr))
    g_o = np.max(librosa.onset.onset_strength(y=c_game_percussive, sr=sr))
    
    # Single wide-range piptrack call, filtered views feed both prosody and pitch harvest below
    pitches, mags = librosa.piptrack(y=c_mic, sr=sr, fmin=75.0, fmax=4000.0)

    # --- NEW: Pitch Harvesting ---
    pitch_values = [pitches[mags[:, t].argmax(), t] for t in range(pitches.shape[1]) if pitches[mags[:, t].argmax(), t] > 0]
    c_peak = max(pitch_values) if pitch_values else 0.0
    c_avg = np.mean(pitch_values) if pitch_values else 0.0

    # Prosody uses only the speech-range pitches (80-400Hz), no second piptrack needed
    pitch_values_speech = [p for p in pitch_values if 80.0 <= p <= 400.0]
    pros = calculate_prosody(pitch_values_speech)
    
    s_time = 0.0
    f_per_sec = pitches.shape[1] // seg_len
    if f_per_sec > 0:
        for j in range(seg_len):
            start_f = j * f_per_sec
            end_f = (j + 1) * f_per_sec if j < seg_len - 1 else pitches.shape[1]
            if np.max(pitches[:, start_f:end_f]) > s_thresh: s_time += 1.0
        
    # Return updated to pass back c_peak and c_avg, and g_crest instead of m_c
    return (i, v_game, v_mic, g_c, g_crest, g_o, pros, (s_time >= s_min), c_peak, c_avg)

# ==========================================
# 4. THE MAIN ENGINE
# ==========================================

def process_vod():
    # --- SINGLE INSTANCE FAILSAFE ---
    # Attempts to bind to a local invisible port. If it fails, another instance is running.
    # The OS automatically releases the port when the script closes or crashes.
    acquire_instance_lock()

    print("\n" + "="*70)
    print(" 🎬  AUTO-VOD MASTER ENGINE (GOLD POCKET MATH)  🎬")
    print("="*70 + "\n")
    
    cfg = load_or_create_config()
    start_t = time.time()
    
    input_video = cfg.get('PATHS', 'input_video')
    highlights_out = cfg.get('PATHS', 'highlights_output')
    montage_out = cfg.get('PATHS', 'montage_output')
    thumb_dir = cfg.get('PATHS', 'thumbnail_folder')
    
    mic_idx = cfg.getint('OBS_TRACKS', 'mic_track')
    game_idx = cfg.getint('OBS_TRACKS', 'game_track')
    
    # === UPDATED THRESHOLDS WITH ARTIFACT CEILINGS ===
    max_h_mins = cfg.getint('THRESHOLDS', 'highlight_max_minutes')
    ex_gate = cfg.getfloat('THRESHOLDS', 'highlight_exception_score')
    target_ratio = cfg.getfloat('THRESHOLDS', 'vod_normal_ratio')
    vod_base_score = cfg.getfloat('THRESHOLDS', 'vod_keep_base_score')
    
    # Scream Logic Reverted to 2000Hz default
    s_thresh = cfg.getfloat('THRESHOLDS', 'scream_threshold_hz')
    s_min = cfg.getfloat('THRESHOLDS', 'scream_min_seconds')
    scream_ceiling = 3500.0 # Anything above this is likely digital noise, not a human scream.
    
    # Linguistic Logic
    panic_wpm = cfg.getfloat('THRESHOLDS', 'panic_wpm_threshold')
    wpm_ceiling = 600.0 # Guinness World Record is ~600. Anything higher is an AI loop.
    
    t_spread = cfg.getfloat('THRESHOLDS', 'thumbnail_spread')
    clip_pad = cfg.getfloat('THRESHOLDS', 'clip_buffer_seconds')
    
    # NEW GUI LINKAGE FOR SMART MERGE:
    cfg_max_len = cfg.getfloat('THRESHOLDS', 'max_clip_length')
    cfg_max_gap = cfg.getfloat('THRESHOLDS', 'max_merge_gap')
    
    b_word = cfg.get('BOOKMARK', 'codeword', fallback='Pineapple').lower().strip()
    h_list = [w.strip().lower() for w in cfg.get('VOCAB', 'hype_words').split(',') if w.strip()]

    # Pre-compile all regex patterns once to get reused across every scored chunk
    hype_patterns    = [re.compile(rf'\b{re.escape(w)}\b') for w in h_list]
    laugh_pattern    = re.compile(r'\b(haha|hahaha|lmao|hehe)\b|\[laughs\]|\(laughing\)')
    bookmark_pattern = re.compile(rf'\b{re.escape(b_word)}\b')

    if not os.path.exists(input_video):
        print(f"[CRITICAL] Input video '{input_video}' not found.")
        sys.exit(1)

    # --- 0. METADATA AUTO-SCANNER ---
    logging.info(f" [*] PROBING: Analyzing container metadata for '{os.path.basename(input_video)}'...")
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', input_video]
    probe_proc = subprocess.Popen(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    probe_out, _ = probe_proc.communicate()
    # Count how many lines/streams were found
    stream_count = len([l for l in probe_out.decode().strip().split('\n') if l]) if probe_out else 0
    logging.info(f" [*] SUCCESS: Found {stream_count} audio streams in file.")

    def extract_to_ram(track_index, name):
        video_dir = os.path.dirname(os.path.abspath(input_video))
        external_file = os.path.join(video_dir, f"{name.lower()}.wav")

        def stream_pcm_to_ram(input_src, map_idx):
            beat.start(f"Streaming {name} to RAM")
            # CHANGED: -f f32le streams raw 32-bit float math, bypassing WAV limits
            cmd = [
                'ffmpeg', '-y', '-i', input_src, '-map', map_idx,
                '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'
            ]
            # Devnull stderr prevents FFmpeg log buffer overflow lockups
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            chunks = []
            # Read in 10MB chunks to bypass OS pipe limits & prevent RAM fragmentation
            while True:
                chunk = proc.stdout.read(10485760)
                if not chunk:
                    break
                # Instantly convert bytes to numpy array, skipping soundfile library
                chunks.append(np.frombuffer(chunk, dtype=np.float32))

            proc.wait()
            beat.stop()

            if chunks:
                return np.concatenate(chunks), 16000
            return None

        # --- 1. EXTERNAL FILE CHECK (Tier 1) ---
        if os.path.exists(external_file):
            logging.info(f" [*] SUCCESS: Found external '{name.lower()}.wav'. Prioritizing for {name}.")
            result = stream_pcm_to_ram(external_file, '0:a:0')
            if result is not None and len(result[0]) > 1000:
                return result

        # --- 2. VIDEO TRACK EXTRACTION (Tier 2) ---
        ff_idx = track_index - 1 
        if ff_idx < stream_count:
            logging.info(f" [*] ATTEMPTING: Extracting isolated {name} from Stream Index {ff_idx} (Track {track_index}).")
            result = stream_pcm_to_ram(input_video, f'0:a:{ff_idx}')
            if result is not None and len(result[0]) > 1000:
                logging.info(f" [*] SUCCESS: {name} Isolated successfully in RAM.")
                return result

        # --- 3. ABSOLUTE FALLBACK (Tier 3) ---
        logging.warning(f" [!] FAILED: {name} Track missing. FALLBACK: Mapping {name} to Master Mix.")
        result = stream_pcm_to_ram(input_video, '0:a:0')
        if result is not None:
            return result
            
        # Ultimate failsafe
        logging.critical(f" [!!!] CRITICAL: All audio extraction tiers failed for '{name}'. No audio data available.")
        return np.array([]), 16000

    y_mic, sr = extract_to_ram(mic_idx, "Mic")
    y_game, _ = extract_to_ram(game_idx, "Game")
    min_samples = min(len(y_mic), len(y_game))
    y_mic = y_mic[:min_samples]
    y_game = y_game[:min_samples]
    total_sec = min_samples // sr
    seg_len = 3
    
    chunk_args = [(i, y_mic[i*sr:(i+seg_len)*sr], y_game[i*sr:(i+seg_len)*sr], sr, seg_len, s_thresh, s_min) for i in range(0, total_sec, seg_len)]
    acoustic_results = {}
    
    # --- NEW: Global trackers for pitch diagnostics ---
    global_pitch_peaks = []
    global_pitch_avgs = []

    max_workers = min(os.cpu_count() or 8, len(chunk_args))
    print(f"\n>>> Acoustic Pass (Pass 1 - {max_workers} Core Multiprocessing)...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(analyze_acoustic_chunk, chunk_args), total=len(chunk_args), desc="Librosa Math", unit="chunk"):
            i, v_g, v_m, g_c, g_crest, g_o, pros, is_s, c_peak, c_avg = res

            # --- SANITIZED DIAGNOSTIC TRACKING (V1.2.1 TWIN-BEAM) ---
            # Gate for Scream Peaks: Only look at the Scream Sanctuary
            # We look for highlights only in the 2200Hz - 3500Hz range.
            if s_thresh < c_peak < scream_ceiling:
                global_pitch_peaks.append(c_peak)

            # Gate for Vocal Baseline: Only look at the Human Speech Floor
            # Humans baseline between 80Hz and 400Hz. This deletes the 521Hz hiss.
            if 80 < c_avg < 400:
                global_pitch_avgs.append(c_avg)

            acoustic_results[i] = {'v_g': v_g, 'v_m': v_m, 'g_c': g_c, 'g_crest': g_crest, 'g_o': g_o, 'pros': pros, 'is_s': is_s, 'c_peak': c_peak, 'c_avg': c_avg}

    del chunk_args  # free the view list
    del y_game      # now free the actual 1.7GB game audio buffer

    # --- PHASE 2: SEMANTIC PASS (WHISPER AI) ---
    # Importing the WhisperModel here ensures it only loads when needed and not during GUI initialization.
    from faster_whisper import WhisperModel
    
    # NATIVE CPU ENFORCEMENT (Bypasses CUDA/PyTorch Thread Thrashing)
    print(f"\n>>> Semantic Pass (Pass 2 - {os.cpu_count() or 12} Thread AI Engine on CPU)...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 12)
    raw_segments = []
    total_hype, peak_wpm, b_count = 0, 0, 0
    wpm_log = [] # Master bucket for average WPM calculation

    # 1. MACRO-PASS: Transcribe the entire microphone track at once (Restores 30-sec context window)
    print("") # Formatting buffer
    # BUG FIX: word_timestamps=True forces the AI to output exact timing for every word
    segs, info = model.transcribe(y_mic, language="en", vad_filter=True, word_timestamps=True)
    
    all_speech_segments = []
    
    # Restored GUI: Dynamic Progress Bar based on transcribed audio timestamps
    with tqdm(total=total_sec, desc="Whisper AI (Full Pass)", unit="sec") as pbar:
        for s in segs:
            all_speech_segments.append(s)
            
            # Calculate how far into the audio the AI currently is
            current_progress = int(s.end)
            if current_progress > pbar.n:
                pbar.update(current_progress - pbar.n)
                
        # Snap the bar to 100% when the generator finishes
        if pbar.n < total_sec:
            pbar.update(total_sec - pbar.n)
            
    logging.info(f" [*] SUCCESS: Transcribed {len(all_speech_segments)} distinct speech segments natively.")
    del y_mic  # no longer needed, free before scoring loop

    # Pre-filter hallucinations once so the scoring loop never re-checks them
    valid_segs = [s for s in all_speech_segments if s.compression_ratio <= 2.4 and s.no_speech_prob <= 0.6]

    # 2. MICRO-PASS: Map AI timestamps back to the 3-second acoustic chunks
    # Pointer-based O(n) scan: valid_segs are in time order, so we never restart from 0
    seg_ptr = 0
    for i in tqdm(range(0, total_sec, seg_len), desc="Chaos Scoring", unit="chunk"):
        data = acoustic_results[i]
        if data['v_m'] < 0.01 and data['v_g'] < 0.01: continue

        chunk_start = i
        chunk_end = i + seg_len
        chunk_texts = []
        speech_overlap = 0.0

        # Advance base pointer past segments that have fully ended before this chunk
        while seg_ptr < len(valid_segs) and valid_segs[seg_ptr].end <= chunk_start:
            seg_ptr += 1

        # Walk forward from the base pointer for segments overlapping this chunk
        j = seg_ptr
        while j < len(valid_segs) and valid_segs[j].start < chunk_end:
            s = valid_segs[j]

            # BUG FIX: Word-level slicing. Only append the exact words spoken inside this 3-second block
            if s.words:
                for w in s.words:
                    # w.start ensures the word is only assigned to ONE chunk, killing the duplication
                    if chunk_start <= w.start < chunk_end:
                        # --- 2. WORD-LEVEL CONFIDENCE GATE ---
                        # Only accept the word if Whisper is at least 40% sure it was actually spoken
                        if w.probability > 0.40:
                            chunk_texts.append(w.word.strip())

            # Calculate the exact duration of the speech that fell into this chunk
            overlap_s = max(s.start, chunk_start)
            overlap_e = min(s.end, chunk_end)
            speech_overlap += (overlap_e - overlap_s)
            j += 1

        text = " ".join(chunk_texts)
        
        # --- ARTIFACT FILTERING ---
        raw_wpm = (len(text.split()) / seg_len) * 60
        # If WPM is impossible, we treat it as 0 for stats/scoring
        valid_wpm = raw_wpm if raw_wpm <= wpm_ceiling else 0.0
        is_panic_mode = (valid_wpm > panic_wpm)

        if valid_wpm > 0: 
            wpm_log.append(valid_wpm)
            if valid_wpm > peak_wpm: peak_wpm = int(valid_wpm)
        
        # --- TWIN-BEAM SCOURING (V1.2.3 TRUE PITCH vs CENTROID) ---
        # Narrowing focus: Mic uses True Pitch (c_peak/c_avg), Game uses Centroid (g_c)
        c_peak_raw = data['c_peak']
        c_avg_raw = data['c_avg']
        g_centroid_raw = data['g_c']
        
        # 1. MIC PITCH: Prevents hardware ringing from triggering Ghost Screams using fundamental pitch.
        is_scream_valid = data['is_s'] and (s_thresh < c_peak_raw < scream_ceiling)
        
        # Filtered Mic Pitch: Only pass valid human talking (80-400Hz) or screaming ranges.
        if (80 < c_avg_raw < 400) or (s_thresh < c_peak_raw < scream_ceiling):
            f_mic_pitch = c_peak_raw
        else:
            f_mic_pitch = 0.0 # Discard digital hiss and edge artifacts
            
        # 2. GAME CENTROID: Focus on high-frequency action (Gunshots/Explosions/Clashes).
        if 1500 < g_centroid_raw < 8000:
            f_g_centroid = g_centroid_raw
        else:
            f_g_centroid = 0.0 # Discard lobby background noise and low rumbles

        # SINGLE CLEAN CALL: Calculate scores using separated Pitch and Centroid systems
        speech_ratio = min(1.0, speech_overlap / seg_len) if speech_overlap > 0 else 0.0
        
        score, h_inc, is_b = calculate_chaos_score(
            data['v_g'], f_g_centroid, data['g_o'], data['g_crest'],
            data['v_m'], f_mic_pitch, is_scream_valid,
            is_panic_mode, data['pros'], text,
            hype_patterns, laugh_pattern, bookmark_pattern, ex_gate, speech_ratio
        )
        
        total_hype += h_inc
        if score > 0:
            raw_segments.append({'start': i, 'end': i+seg_len, 'score': score, 'bookmarked': is_b})
            if is_b: b_count += 1

    # --- PHASE 2.5: STRICT SMART MERGE ENGINE ---
    def merge(segs, pad, max_gap=15.0, max_len=300.0, chunk_sec=3):
        if not segs: return []
        segs.sort(key=lambda x: x['start'])
        merged_list = []
        current_clip = dict(segs[0])
        current_clip['score_sum'] = current_clip['score'] # Track sum for density math

        for nxt in segs[1:]:
            gap = nxt['start'] - current_clip['end']
            potential_length = nxt['end'] - current_clip['start']

            # RULE: Merge ONLY if gap is small AND the resulting clip stays under max_len
            if gap <= max_gap and potential_length <= max_len:
                current_clip['end'] = max(current_clip['end'], nxt['end'])
                current_clip['score_sum'] += nxt['score'] # Add to the density pool
                current_clip['bookmarked'] = current_clip['bookmarked'] or nxt['bookmarked']
            else:
                # Close the current clip and calculate its final density score
                actual_length = current_clip['end'] - current_clip['start']
                current_clip['score'] = current_clip['score_sum'] / (actual_length / chunk_sec) if actual_length > 0 else current_clip['score']
                merged_list.append(current_clip)
                
                current_clip = dict(nxt)
                current_clip['score_sum'] = current_clip['score']
        
        # Close the final clip
        actual_length = current_clip['end'] - current_clip['start']
        current_clip['score'] = current_clip['score_sum'] / (actual_length / chunk_sec) if actual_length > 0 else current_clip['score']
        merged_list.append(current_clip)

        # Apply padding and ensure no overlaps
        for s in merged_list:
            s['start'] = max(0.0, s['start'] - pad)
            s['end'] = s['end'] + pad

        # Overlap Protection: If padding causes clips to touch, snap them to the midpoint
        for i in range(1, len(merged_list)):
            if merged_list[i]['start'] < merged_list[i-1]['end']:
                midpoint = (merged_list[i-1]['end'] + merged_list[i]['start']) / 2.0
                merged_list[i-1]['end'] = merged_list[i]['start'] = midpoint
        
        return merged_list

    # --- PHASE 3: AGGRESSIVE FILTER & CONSOLIDATION ---
    valid_chunks = [s for s in raw_segments if s['score'] > 2.0 or s['bookmarked']]
    
    # The merge function enforces cfg_max_len natively, guaranteeing organic scores
    # and preventing highlight score cloning.
    merged_clips = merge(valid_chunks, clip_pad, max_gap=cfg_max_gap, max_len=cfg_max_len, chunk_sec=seg_len)
    
    # --- PHASE 4: TRIMMED VOD LOGIC (RATIO PRUNING) ---
    max_vod_sec = total_sec * target_ratio
    
    # STEP 1: The Absolute Floor (Your intended logic)
    # Instantly delete any clip below the base score (unless it has a Yam bookmark)
    final_vod = [s for s in merged_clips if s['score'] >= vod_base_score or s.get('bookmarked')]
    
    # STEP 2: The Ratio Pruner
    # If surviving clips STILL exceed the target duration, prune the lowest remaining scores
    prunable = sorted([s for s in final_vod if not s.get('bookmarked')], key=lambda x: x['score'])
    
    current_vod_sec = sum(s['end'] - s['start'] for s in final_vod)
    while current_vod_sec > max_vod_sec and prunable:
        removed_clip = prunable.pop(0)
        final_vod.remove(removed_clip)
        current_vod_sec -= (removed_clip['end'] - removed_clip['start'])

    # --- PHASE 5: HIGHLIGHT REEL BUCKET LOGIC ---
    max_h_sec = max_h_mins * 60
    
    # NEW TIE-BREAKER LOGIC: Sorts by Score first. If Scores tie, it picks randomly.
    ranked = sorted(merged_clips, key=lambda x: (x['score'], random.random()), reverse=True)
    
    final_h = []
    curr_dur = 0
    for s in ranked:
        dur = s['end'] - s['start']
        # Fill bucket highest-score first. Overfill ONLY for Yam exceptions or exceptionally high scores (ex_gate).
        if curr_dur + dur <= max_h_sec or s['bookmarked'] or s['score'] >= ex_gate:
            final_h.append(s)
            curr_dur += dur

    # --- PHASE 5.2: SHORTS & THUMBNAIL GENERATION ---
    num_shorts = cfg.getint('SHORTS', 'num_shorts', fallback=5)
    if num_shorts > 0:
        print("\n>>> Slicing Top-Tier Shorts & Extracting Thumbnails...")
        beat.start("Exporting Shorts & Media")
        # Passing t_spread and thumb_dir down into the module
        shorts_module.generate_shorts(ranked, input_video, num_shorts, t_spread, thumb_dir)
        beat.stop()

    def render(segs, out_file, desc):
        if not segs: return
        cp = os.path.join(SCRIPT_DIR, "concat_list_temp.txt")
        with open(cp, "w") as f:
            for c in sorted(segs, key=lambda x: x['start']): f.write(f"file '{input_video}'\ninpoint {c['start']}\noutpoint {c['end']}\n")
        beat.start(f"Rendering {desc}")
        try:
            subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', cp, '-c:v', 'copy', '-c:a', 'aac', '-af', 'aresample=async=1', out_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            os.remove(cp)
            beat.stop()
        
    print("\n>>> 4/4: Lossless Video Rendering...")
    render(final_vod, montage_out, "Trimmed VOD")
    render(final_h, highlights_out, "Highlight Reel")

    end_t = time.time()
    elapsed = end_t - start_t
    trim_min = (total_sec - sum(s['end'] - s['start'] for s in final_vod)) / 60
    
    print("\n" + "="*70)
    print(f" 🏁  PROCESS COMPLETE  🏁")
    print("="*70)
    print(f" ⏱️  Processing Time: {elapsed / 60:.1f} minutes")
    
    # --- DIAGNOSTIC FIX: Show actual clips generated vs clips kept ---
    print(f" 📦  Total Raw Clips Generated (Pre-Pruning): {len(merged_clips)}")
    print(f" 🎞️  Clips Kept in Trimmed VOD: {len(final_vod)}")
    
    print(f" ✂️  Dead Air Trimmed: {int(trim_min)} minutes")
    print(f" 🔥  Total Hype Words Found: {total_hype}")
    print(f" 🚀  Peak WPM Found: {peak_wpm}")
    
    avg_wpm = int(sum(wpm_log) / len(wpm_log)) if wpm_log else 0
    print(f" 📊  Average WPM (When Speaking): {avg_wpm}")
    
    # --- NEW: Pitch Diagnostic Output ---
    overall_peak_pitch = max(global_pitch_peaks) if global_pitch_peaks else 0
    overall_avg_pitch = int(np.mean(global_pitch_avgs)) if global_pitch_avgs else 0
    print(f" 🗣️  Average Voice Pitch (Baseline): {overall_avg_pitch} Hz")
    print(f" 😱  Peak Mic Frequency (Scream Max): {int(overall_peak_pitch)} Hz")
    
    print(f" 🏆  Highest Chaos Score Found: {max([s['score'] for s in merged_clips] if merged_clips else [0]):.1f}")
    print("\n --- TOP 5 HIGHLIGHT EVENTS ---")
    top_clips = sorted(merged_clips, key=lambda x: x['score'], reverse=True)[:5]
    for idx, c in enumerate(top_clips):
        m, sec = divmod(c['start'], 60)
        print(f"  {idx+1}. Score {c['score']:.1f} @ {int(m):02d}:{int(sec):02d}")

    print("\n --- LOWEST 5 CLIPS KEPT (Trimmed VOD) ---")
    if final_vod:
        bottom_clips = sorted(final_vod, key=lambda x: x['score'])[:5]
        for idx, c in enumerate(bottom_clips):
            m, sec = divmod(c['start'], 60)
            print(f"  {idx+1}. Score {c['score']:.1f} @ {int(m):02d}:{int(sec):02d}")
    else:
        print("  No clips survived the pruning process.")
    
    print(f"\n 🔖  Bookmarks: {b_count} Yams found")
    print("="*70)
    # The function now officially ends here, triggering Python's Garbage Collector.

if __name__ == "__main__":
    process_vod()
    # The terminal stays open, but the 2GB of RAM is instantly freed back to Windows.
    input("\nPress Enter to exit...")
