import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import filedialog
import configparser
import os
import sys
import subprocess

# ==========================================
# SYSTEM PATHING
# ==========================================
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.ini")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AutoVodGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Auto-VOD Editor")
        self.geometry("720x800")
        self.resizable(True, True)

        self.config = configparser.ConfigParser()
        self.load_config()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) 

        # --- HEADER ---
        self.header = ctk.CTkLabel(self, text="Auto-VOD Creator", font=ctk.CTkFont(size=24, weight="bold"))
        self.header.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # --- SCROLLABLE SETTINGS ---
        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.scroll.grid_columnconfigure(1, weight=1)

        row_idx = 0

        # === SECTION: PATHS & OBS TRACKS ===
        ctk.CTkLabel(self.scroll, text="PATHS & OBS TRACKS", font=ctk.CTkFont(size=16, weight="bold"), text_color="#3498db").grid(row=row_idx, column=0, columnspan=3, pady=(10, 5), sticky="w")
        row_idx += 1

        ctk.CTkLabel(self.scroll, text="Input Video:").grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
        self.file_entry = ctk.CTkEntry(self.scroll)
        self.file_entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        self.file_entry.insert(0, self.config.get('PATHS', 'input_video', fallback=''))
        
        self.browse_btn = ctk.CTkButton(self.scroll, text="Browse", width=60, command=self.browse_file)
        self.browse_btn.grid(row=row_idx, column=2, padx=(0, 10), pady=5)
        row_idx += 1

        ctk.CTkLabel(self.scroll, text="Mic / Game Tracks (OBS Index):").grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
        self.track_frame = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.track_frame.grid(row=row_idx, column=1, columnspan=2, sticky="w", padx=5)
        
        self.mic_entry = ctk.CTkEntry(self.track_frame, width=50)
        self.mic_entry.insert(0, self.config.get('OBS_TRACKS', 'mic_track', fallback='2'))
        self.mic_entry.pack(side="left", padx=(0, 10))
        
        self.game_entry = ctk.CTkEntry(self.track_frame, width=50)
        self.game_entry.insert(0, self.config.get('OBS_TRACKS', 'game_track', fallback='3'))
        self.game_entry.pack(side="left")
        row_idx += 1

        # --- SLIDER HELPER ---
        self.sliders = {}
        def add_slider(section, label_text, config_key, from_val, to_val, default_val, is_float=True, steps=None):
            nonlocal row_idx
            val = float(self.config.get(section, config_key, fallback=str(default_val)))
            lbl = ctk.CTkLabel(self.scroll, text=f"{label_text}: {val:.2f}" if is_float else f"{label_text}: {int(val)}")
            lbl.grid(row=row_idx, column=0, padx=10, pady=(15, 0), sticky="w")
            
            slider = ctk.CTkSlider(self.scroll, from_=from_val, to=to_val, number_of_steps=steps)
            slider.set(val)
            slider.grid(row=row_idx+1, column=0, columnspan=3, padx=10, pady=(0, 5), sticky="ew")
            
            def update_lbl(v, l=lbl, t=label_text, f=is_float):
                l.configure(text=f"{t}: {v:.2f}" if f else f"{t}: {int(v)}")
            
            slider.configure(command=update_lbl)
            self.sliders[config_key] = (slider, is_float, section)
            row_idx += 2

        # === SECTION: VOD TRIMMING LOGIC ===
        ctk.CTkLabel(self.scroll, text="VOD TRIMMING LOGIC", font=ctk.CTkFont(size=16, weight="bold"), text_color="#3498db").grid(row=row_idx, column=0, columnspan=3, pady=(20, 5), sticky="w")
        row_idx += 1
        add_slider('THRESHOLDS', 'VIP Keep Base Score', 'vod_keep_base_score', 2, 30, 7, False, 28)
        add_slider('THRESHOLDS', 'Target VOD Ratio (%)', 'vod_normal_ratio', 0.1, 1.0, 0.75, True, 18)

        # === SECTION: HIGHLIGHTS & THUMBNAILS ===
        ctk.CTkLabel(self.scroll, text="HIGHLIGHTS & THUMBNAILS", font=ctk.CTkFont(size=16, weight="bold"), text_color="#3498db").grid(row=row_idx, column=0, columnspan=3, pady=(20, 5), sticky="w")
        row_idx += 1
        add_slider('THRESHOLDS', 'Highlight Max Minutes', 'highlight_max_minutes', 1, 120, 55, False, 119)
        add_slider('THRESHOLDS', 'Thumbnail Spread (sec)', 'thumbnail_spread', 1.0, 10.0, 6.0, True, 18)

        # === SECTION: VOCAB & MANUAL BOOKMARKING ===
        ctk.CTkLabel(self.scroll, text="VOCAB & MANUAL BOOKMARKING", font=ctk.CTkFont(size=16, weight="bold"), text_color="#e67e22").grid(row=row_idx, column=0, columnspan=3, pady=(20, 5), sticky="w")
        row_idx += 1
        ctk.CTkLabel(self.scroll, text="Bookmark Codeword (Yams Override):").grid(row=row_idx, column=0, padx=10, sticky="w")
        self.bookmark_entry = ctk.CTkEntry(self.scroll)
        self.bookmark_entry.grid(row=row_idx, column=1, columnspan=2, padx=10, pady=5, sticky="ew")
        self.bookmark_entry.insert(0, self.config.get('BOOKMARK', 'codeword', fallback='Pineapple'))
        row_idx += 1
        
        ctk.CTkLabel(self.scroll, text="Hype Words (comma separated):").grid(row=row_idx, column=0, padx=10, sticky="w")
        row_idx += 1
        self.vocab_box = ctk.CTkTextbox(self.scroll, height=80)
        self.vocab_box.grid(row=row_idx, column=0, columnspan=3, padx=10, pady=(0, 20), sticky="ew")
        # UPDATED: Matches the engine's refined action vocabulary
        self.vocab_box.insert("1.0", self.config.get('VOCAB', 'hype_words', fallback='clip that, oh fuck, oh shit, reloading, grenade'))

        # === SECTION: SHORTS GENERATOR ===
        ctk.CTkLabel(self.scroll, text="SHORTS GENERATOR", font=ctk.CTkFont(size=16, weight="bold"), text_color="#9b59b6").grid(row=row_idx, column=0, columnspan=3, pady=(20, 5), sticky="w")
        row_idx += 1
        add_slider('SHORTS', 'Number of Shorts to Export', 'num_shorts', 0, 20, 10, False, 20)

        # --- FOOTER ---
        self.run_btn = ctk.CTkButton(self, text="SAVE SETTINGS & RUN MASTER ENGINE", height=60, font=ctk.CTkFont(size=16, weight="bold"), fg_color="#2ecc71", hover_color="#27ae60", command=self.save_and_run)
        self.run_btn.grid(row=2, column=0, padx=20, pady=20, sticky="ew")

    def load_config(self):
        self.config.read(CONFIG_FILE)
        for section in ['PATHS', 'OBS_TRACKS', 'THRESHOLDS', 'VOCAB', 'BOOKMARK', 'SHORTS']:
            if not self.config.has_section(section): 
                self.config.add_section(section)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.mov *.avi")])
        if file_path:
            full_path = os.path.abspath(file_path)
            self.file_entry.delete(0, 'end')
            self.file_entry.insert(0, full_path)

    def save_and_run(self):
        input_path = self.file_entry.get().strip()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid input video file.")
            return

        self.config.set('PATHS', 'input_video', input_path)
        self.config.set('OBS_TRACKS', 'mic_track', self.mic_entry.get().strip())
        self.config.set('OBS_TRACKS', 'game_track', self.game_entry.get().strip())
        self.config.set('VOCAB', 'hype_words', self.vocab_box.get("1.0", "end").strip())
        self.config.set('BOOKMARK', 'codeword', self.bookmark_entry.get().strip())

        for key, data in self.sliders.items():
            slider_obj, is_float, section = data
            val = slider_obj.get()
            self.config.set(section, key, f"{val:.2f}" if is_float else str(int(val)))

        with open(CONFIG_FILE, 'w') as f:
            self.config.write(f)

        script_path = os.path.join(SCRIPT_DIR, "vod_trimmer_master.py")
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Engine file not found: {script_path}")
            return

        if os.name == 'nt':
            subprocess.Popen(['cmd.exe', '/k', 'python', script_path], creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=SCRIPT_DIR)
        else:
            subprocess.Popen(['python3', script_path], cwd=SCRIPT_DIR)
            
        self.destroy()

if __name__ == "__main__":
    app = AutoVodGUI()
    app.mainloop()
