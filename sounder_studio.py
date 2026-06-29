#!/usr/bin/env python3
"""
Sounder Studio - Advanced Image-Based Audio Editor & Synthesizer
Creative spectral manipulation.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, simpledialog
import numpy as np
import threading
import math
import random
from threading import Lock

# Core imports
import librosa
import soundfile as sf
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance
import scipy.ndimage
import scipy.signal

# Audio playback (optional - for systems with PortAudio)
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: Audio playback not available ({e})")
    print("The application will run but audio playback features will be disabled")
    AUDIO_AVAILABLE = False
    # Create dummy sd module for import compatibility
    class DummySD:
        class OutputStream:
            def __init__(self, *args, **kwargs):
                raise OSError("Audio playback not available")
        class CallbackStop(Exception):
            pass
    sd = DummySD()

class Tool:
    """Tool types"""
    SELECT = "Select"
    BRUSH = "Brush"
    ERASER = "Eraser"
    LINE = "Line"
    FILL = "Fill"
    PARTICLE_SPRAY = "Particle Spray"
    WARP_MARKER = "Warp Marker"
    SPECTRAL_SHAPER = "Spectral Shaper"
    GRANULAR = "Granular"
    FREEZE = "Freeze"

class SelectionMode:
    """Selection modes"""
    RECTANGLE = "Rectangle"
    LASSO = "Lasso"

class MorphState:
    """Morphing state storage"""
    def __init__(self, image, name="State"):
        self.image = image.copy()
        self.name = name

class SounderStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Sounder Studio By Joel Lagace  - Creative Spectral Editor")
        
        # Responsive sizing based on screen resolution
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Base dimensions (70% of screen to leave room for taskbars)
        init_w = int(screen_width * 0.70)
        init_h = int(screen_height * 0.70)
        
        # Constrain bounds, strictly limiting height so it never clips under the taskbar
        init_w = max(900, min(init_w, 1600))
        init_h = max(600, min(init_h, screen_height - 80))
        
        # Calculate coordinates to center the window and shift it slightly up
        x = int((screen_width - init_w) / 2)
        y = max(0, int((screen_height - init_h) / 2) - 40)
        
        # Apply size and position: Width x Height + X_offset + Y_offset
        self.root.geometry(f"{init_w}x{init_h}+{x}+{y}")
        self.root.minsize(900, 600)

        # Audio Settings
        self.sample_rate = 44100
        self.n_fft = 2048
        self.hop_length = 512

        # State Variables
        self.current_audio = None
        self.audio_duration = 0
        self.is_playing = False
        self.tk_image = None
        self.original_image = None
        self.modified_image = None
        self.undo_stack = []
        self.redo_stack = []

        # Live Playback State
        self.stream = None
        self.audio_index = 0.0  # Always a float, accessed from both threads
        self.audio_lock = Lock()  # Thread safety for audio_index

        # Canvas Settings - Initial values, updated dynamically via Configure event
        self.canvas_width = 800
        self.canvas_height = 400
        self._resize_timer = None

        # Tool State
        self.current_tool = Tool.SELECT
        self.selection_mode = SelectionMode.RECTANGLE
        self.brush_size = 5
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.selection_start = None
        self.selection_coords = None
        self.selection_lasso_points = []
        self.active_selection = None
        self.warp_markers = []  # List of (x, time_stretch_factor)

        # Granular State
        self.grains = []  # List of grain parameters
        self.grain_density = 5
        self.grain_size = 100
        self.grain_pitch = 1.0

        # Morphing States
        self.morph_states = []
        self.current_morph_value = 0.0

        # Effect Settings
        self.brightness = 0
        self.contrast = 1.0
        self.invert_enabled = False
        self.tint_color = None
        self.freeze_active = False
        self.freeze_buffer = None

        # Particle Spray
        self.particle_color = 255
        self.particle_spread = 10
        self.particle_density = 3

        # Spectral Shaper
        self.spectral_curve = []  # List of (frequency, amplitude) points

        # Automation
        self.automation_recording = False
        self.automation_data = []

        # Audio Buffers for effects
        self.grain_buffer = []

        self.setup_ui()
        self.setup_canvas_events()
        self.update_status("🎵 Sounder Studio Ready - Load an image to begin creating!")

    def setup_ui(self):
        """Build the complete UI with tabbed panels"""
        # Main container
        main_container = tk.Frame(self.root, bg="#2b2b2b")
        main_container.pack(fill=tk.BOTH, expand=True)

        # === TOP BAR ===
        top_bar = tk.Frame(main_container, bg="#3c3c3c", relief=tk.RAISED, bd=1)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(top_bar, text="🎵 Studio", font=("Arial", 12, "bold"),
                bg="#3c3c3c", fg="#00ff00").pack(side=tk.LEFT, padx=5, pady=5)

        # File Operations
        file_frame = tk.Frame(top_bar, bg="#3c3c3c")
        file_frame.pack(side=tk.LEFT, padx=5)
        tk.Button(file_frame, text="📁 Load", command=self.load_image_and_process,
                 bg="#4c4c4c", fg="white", width=6).pack(side=tk.LEFT, padx=1)
        tk.Button(file_frame, text="💾 Save", command=self.save_image,
                 bg="#4c4c4c", fg="white", width=6).pack(side=tk.LEFT, padx=1)
        tk.Button(file_frame, text="🎶→WAV", command=self.image_to_wav,
                 bg="#4c4c4c", fg="white", width=7).pack(side=tk.LEFT, padx=1)
        tk.Button(file_frame, text="WAV→🎵", command=self.wav_to_image,
                 bg="#4c4c4c", fg="white", width=7).pack(side=tk.LEFT, padx=1)
        tk.Button(file_frame, text="🔬 Spectrum", command=self.open_visual_spectrogram,
                 bg="#5c3c6c", fg="white", width=9).pack(side=tk.LEFT, padx=1)

        # Playback Controls
        play_frame = tk.Frame(top_bar, bg="#3c3c3c")
        play_frame.pack(side=tk.LEFT, padx=10)
        self.btn_play = tk.Button(play_frame, text="▶", command=self.play_audio,
                                 fg="green", bg="#2a5a2a", font=("Arial", 12, "bold"), width=3)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        tk.Button(play_frame, text="⏹", command=self.stop_audio,
                 fg="red", bg="#5a2a2a", font=("Arial", 12, "bold"), width=3).pack(side=tk.LEFT, padx=2)
        tk.Button(play_frame, text="⏮", command=self.seek_start,
                 fg="cyan", bg="#2a5a5a", width=3).pack(side=tk.LEFT, padx=2)
        # Global Apply button
        tk.Button(play_frame, text="✓ Apply & Regen", command=self.apply_all_and_regenerate,
                 fg="#00ffff", bg="#2a4a4a", font=("Arial", 9, "bold"), width=14).pack(side=tk.LEFT, padx=5)

        # Speed Control
        speed_frame = tk.Frame(top_bar, bg="#3c3c3c")
        speed_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(speed_frame, text="Spd:", bg="#3c3c3c", fg="white").pack(side=tk.LEFT)
        self.speed_slider = tk.Scale(speed_frame, from_=-10.0, to=10.0, resolution=0.1,
                                    orient=tk.HORIZONTAL, length=80, showvalue=False,
                                    bg="#3c3c3c", fg="white", highlightthickness=0)
        self.speed_slider.set(1.0)
        self.speed_slider.pack(side=tk.LEFT)
        self.speed_label = tk.Label(speed_frame, text="1.0x", width=4, bg="#3c3c3c", fg="cyan")
        self.speed_label.pack(side=tk.LEFT)
        self.speed_slider.config(command=lambda v: self.speed_label.config(text=f"{float(v):.1f}x"))

        # === MAIN CONTENT AREA ===
        content_area = tk.Frame(main_container, bg="#2b2b2b")
        content_area.pack(fill=tk.BOTH, expand=True)

        # === LEFT PANEL: TOOLS ===
        tools_panel = tk.Frame(content_area, bg="#3c3c3c", width=160, relief=tk.RAISED, bd=1)
        tools_panel.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        tools_panel.pack_propagate(False)

        # Tools Header
        tk.Label(tools_panel, text="🔧 TOOLS", font=("Arial", 10, "bold"),
                bg="#3c3c3c", fg="#00ff00").pack(pady=5)

        # Tool Buttons (scrollable)
        tool_canvas = tk.Canvas(tools_panel, bg="#3c3c3c", highlightthickness=0)
        scrollbar = tk.Scrollbar(tools_panel, orient="vertical", command=tool_canvas.yview)
        tool_frame = tk.Frame(tool_canvas, bg="#3c3c3c")

        tool_canvas.configure(yscrollcommand=scrollbar.set)
        tool_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        tool_canvas.create_window((0, 0), window=tool_frame, anchor="nw", width=140)

        def configure_tool_canvas(event):
            tool_canvas.configure(scrollregion=tool_canvas.bbox("all"))
        tool_frame.bind("<Configure>", configure_tool_canvas)

        # Tool buttons
        self.tool_var = tk.StringVar(value=Tool.SELECT)
        tools_list = [
            (Tool.SELECT, "🖱️ Select"),
            (Tool.BRUSH, "🖌️ Brush"),
            (Tool.ERASER, "🧹 Eraser"),
            (Tool.LINE, "📏 Line"),
            (Tool.FILL, "🪣 Fill"),
            (Tool.PARTICLE_SPRAY, "💫 Particles"),
            (Tool.WARP_MARKER, "⏯️ Warp Mark"),
            (Tool.SPECTRAL_SHAPER, "🌊 Spectral"),
            (Tool.GRANULAR, "🎵 Granular"),
            (Tool.FREEZE, "❄️ Freeze"),
        ]

        for tool, label in tools_list:
            rb = tk.Radiobutton(tool_frame, text=label, variable=self.tool_var,
                              value=tool, command=self.set_tool,
                              bg="#3c3c3c", fg="white", selectcolor="#2a5a2a",
                              indicatoron=0, anchor="w")
            rb.pack(pady=1, padx=2, fill=tk.X)

        # Selection Mode
        tk.Label(tool_frame, text="─ Selection ─", bg="#3c3c3c", fg="gray").pack(pady=(10, 2))
        self.selection_var = tk.StringVar(value=SelectionMode.RECTANGLE)
        for mode in [SelectionMode.RECTANGLE, SelectionMode.LASSO]:
            rb = tk.Radiobutton(tool_frame, text=mode, variable=self.selection_var,
                              value=mode, command=self.set_selection_mode,
                              bg="#3c3c3c", fg="white", selectcolor="#2a5a2a",
                              indicatoron=0, anchor="w")
            rb.pack(pady=1, padx=2, fill=tk.X)

        # Brush Size
        tk.Label(tool_frame, text="─ Brush ─", bg="#3c3c3c", fg="gray").pack(pady=(10, 2))
        tk.Label(tool_frame, text="Size:", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.brush_slider = tk.Scale(tool_frame, from_=1, to=100, orient=tk.HORIZONTAL,
                                    bg="#3c3c3c", fg="white", highlightthickness=0)
        self.brush_slider.set(5)
        self.brush_slider.pack(fill=tk.X, padx=2)

        # === CENTER: CANVAS ===
        canvas_container = tk.Frame(content_area, bg="#1a1a1a")
        canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Time ruler
        time_ruler = tk.Canvas(canvas_container, height=20, bg="#2a2a2a", highlightthickness=0)
        time_ruler.pack(fill=tk.X)
        self.time_ruler = time_ruler
        self.draw_time_ruler(time_ruler)

        # Frequency ruler
        freq_container = tk.Frame(canvas_container)
        freq_container.pack(fill=tk.BOTH, expand=True)

        freq_ruler = tk.Canvas(freq_container, width=50, bg="#2a2a2a", highlightthickness=0)
        freq_ruler.pack(side=tk.LEFT, fill=tk.Y)
        self.freq_ruler = freq_ruler
        self.draw_freq_ruler(freq_ruler)

        canvas_frame = tk.Frame(freq_container, bg="black")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Playhead
        self.playhead = self.canvas.create_line(0, 0, 0, self.canvas_height, fill="#ff0000", width=2, state="hidden")

        # Selection elements
        self.selection_rect = self.canvas.create_rectangle(0, 0, 0, 0, outline="yellow", width=2, state="hidden", dash=(5, 5))
        self.selection_lasso = self.canvas.create_line(0, 0, 0, 0, fill="yellow", width=2, state="hidden", smooth=True)

        # Warp markers display
        self.warp_lines = []

        # === RIGHT PANEL: EFFECTS TABS ===
        effects_panel = tk.Frame(content_area, bg="#3c3c3c", width=320, relief=tk.RAISED, bd=1)
        effects_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=2, pady=2)
        effects_panel.pack_propagate(False)

        # Tab buttons
        tab_frame = tk.Frame(effects_panel, bg="#3c3c3c")
        tab_frame.pack(fill=tk.X)

        self.current_tab = "effects"
        tabs = {
            "effects": "🎨 FX",
            "spectral": "📊 Spec",
            "granular": "🎵 Grain",
            "creative": "✨ New",
            "morph": "🔄 Morph"
        }

        for tab_id, label in tabs.items():
            btn = tk.Button(tab_frame, text=label, command=lambda t=tab_id: self.switch_tab(t),
                           bg="#4c4c4c", fg="white", relief=tk.FLAT, padx=2, pady=2, font=("Arial", 9))
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Tab content containers
        self.tab_contents = {}
        for tab_id in tabs.keys():
            container = tk.Frame(effects_panel, bg="#3c3c3c")
            self.tab_contents[tab_id] = container

        self.setup_effects_tab()
        self.setup_spectral_tab()
        self.setup_granular_tab()
        self.setup_creative_tab()
        self.setup_morph_tab()

        # Show default tab
        self.switch_tab("effects")

        # === BOTTOM BAR: STATUS + WAVEFORM ===
        bottom_bar = tk.Frame(main_container, bg="#3c3c3c", relief=tk.RAISED, bd=1)
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        tk.Label(bottom_bar, textvariable=self.status_var, bg="#3c3c3c", fg="#00ff00",
                anchor=tk.W).pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)

        # Undo/Redo buttons
        undo_frame = tk.Frame(bottom_bar, bg="#3c3c3c")
        undo_frame.pack(side=tk.RIGHT, padx=5)
        tk.Button(undo_frame, text="↶", command=self.undo, width=3, bg="#4c4c4c").pack(side=tk.LEFT)
        tk.Button(undo_frame, text="↷", command=self.redo, width=3, bg="#4c4c4c").pack(side=tk.LEFT)

        # Clipboard
        self.clipboard = None

    def draw_time_ruler(self, canvas):
        """Draw time ruler with markers based on actual audio duration"""
        w = max(10, self.canvas_width)
        canvas.delete("all")
        canvas.create_line(0, 15, w, 15, fill="#666", width=1)

        # Get actual duration if available
        if hasattr(self, 'audio_duration') and self.audio_duration > 0:
            duration = self.audio_duration
        else:
            duration = 10.0  # Fallback

        # Draw appropriate number of markers based on duration
        if duration <= 10:
            num_markers = 11
            step = duration / 10
        elif duration <= 30:
            num_markers = 11
            step = duration / 10
        elif duration <= 60:
            num_markers = 7
            step = duration / 6
        else:
            num_markers = 11
            step = duration / 10

        for i in range(num_markers):
            x = (i / (num_markers - 1)) * w
            time_val = i * step
            canvas.create_line(x, 10, x, 15, fill="#999")

            # Format label based on duration
            if duration < 10:
                label = f"{time_val:.1f}s"
            elif duration < 60:
                label = f"{time_val:.0f}s"
            else:
                minutes = int(time_val / 60)
                secs = int(time_val % 60)
                label = f"{minutes}:{secs:02d}"

            canvas.create_text(x, 5, text=label, fill="#888", font=("Arial", 7))

    def draw_freq_ruler(self, canvas):
        """Draw frequency ruler"""
        h = max(10, self.canvas_height)
        canvas.delete("all")
        # Draw main line with proper padding
        canvas.create_line(42, 5, 42, h-5, fill="#666", width=1)
        for i in range(0, 11):
            y = (i / 10) * h
            # Keep labels within canvas bounds with padding
            y = max(10, min(h - 10, y))
            canvas.create_line(42, y, 47, y, fill="#999")
            freq = int(22050 * (1 - i / 10))
            # Position text to be fully visible
            canvas.create_text(25, y, text=f"{freq}Hz", fill="#888", font=("Arial", 7), angle=90)

    def switch_tab(self, tab_id):
        """Switch between effect panels"""
        self.current_tab = tab_id
        for tid, container in self.tab_contents.items():
            if tid == tab_id:
                container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            else:
                container.pack_forget()

    def setup_effects_tab(self):
        """Setup basic effects tab"""
        container = self.tab_contents["effects"]

        # Image Effects
        frame = tk.LabelFrame(container, text="🎨 Image Processing", bg="#3c3c3c", fg="cyan")
        frame.pack(fill=tk.X, pady=5)

        # Brightness
        tk.Label(frame, text="Brightness", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.brightness_slider = tk.Scale(frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                                         bg="#3c3c3c", fg="white", highlightthickness=0)
        self.brightness_slider.set(0)
        self.brightness_slider.pack(fill=tk.X, padx=5)

        # Contrast
        tk.Label(frame, text="Contrast", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.contrast_slider = tk.Scale(frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL,
                                       bg="#3c3c3c", fg="white", highlightthickness=0)
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(fill=tk.X, padx=5)

        # Gamma
        tk.Label(frame, text="Gamma", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.gamma_slider = tk.Scale(frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL,
                                     bg="#3c3c3c", fg="white", highlightthickness=0)
        self.gamma_slider.set(1.0)
        self.gamma_slider.pack(fill=tk.X, padx=5)

        # Invert
        self.invert_var = tk.BooleanVar()
        tk.Checkbutton(frame, text="Invert Colors", variable=self.invert_var,
                      bg="#3c3c3c", fg="white", selectcolor="#2a5a2a",
                      command=self.apply_live_effects).pack(anchor=tk.W, padx=5)

        # Tint
        tk.Button(frame, text="🎨 Choose Tint", command=self.choose_tint,
                 bg="#4c4c4c", fg="white").pack(fill=tk.X, padx=5, pady=2)
        self.tint_button = tk.Label(frame, text="No tint", bg="gray", relief=tk.SUNKEN)
        self.tint_button.pack(fill=tk.X, padx=5)

        # Apply buttons
        btn_frame = tk.Frame(frame, bg="#3c3c3c")
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Apply", command=self.apply_effects_permanent,
                 bg="#2a5a2a", fg="white").pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="Reset", command=self.reset_effects,
                 bg="#5a2a2a", fg="white").pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

    def setup_spectral_tab(self):
        """Setup spectral processing tab"""
        container = self.tab_contents["spectral"]

        # Frequency Operations
        frame = tk.LabelFrame(container, text="📊 Frequency Operations", bg="#3c3c3c", fg="cyan")
        frame.pack(fill=tk.X, pady=5)

        # Vertical Scale (Pitch Shift)
        tk.Label(frame, text="Vertical Scale (Pitch)", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.vertical_scale_slider = tk.Scale(frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL,
                                            bg="#3c3c3c", fg="white", highlightthickness=0)
        self.vertical_scale_slider.set(1.0)
        self.vertical_scale_slider.pack(fill=tk.X, padx=5)

        tk.Button(frame, text="Apply Vertical Scale", command=self.apply_vertical_scale,
                 bg="#4c4c4c", fg="white").pack(fill=tk.X, padx=5, pady=2)

        # Frequency Shift
        tk.Label(frame, text="Frequency Shift (Hz)", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.freq_shift_slider = tk.Scale(frame, from_=-5000, to=5000, orient=tk.HORIZONTAL,
                                         bg="#3c3c3c", fg="white", highlightthickness=0)
        self.freq_shift_slider.set(0)
        self.freq_shift_slider.pack(fill=tk.X, padx=5)

        tk.Button(frame, text="Apply Frequency Shift", command=self.apply_frequency_shift,
                 bg="#4c4c4c", fg="white").pack(fill=tk.X, padx=5, pady=2)

        # Band Filter
        tk.Label(frame, text="Frequency Band", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        filter_frame = tk.Frame(frame, bg="#3c3c3c")
        filter_frame.pack(fill=tk.X, padx=5)

        self.filter_low = tk.IntVar(value=20)
        self.filter_high = tk.IntVar(value=80)
        tk.Entry(filter_frame, textvariable=self.filter_low, width=5, bg="#2a2a2a", fg="white").pack(side=tk.LEFT)
        tk.Label(filter_frame, text="-", bg="#3c3c3c", fg="white").pack(side=tk.LEFT)
        tk.Entry(filter_frame, textvariable=self.filter_high, width=5, bg="#2a2a2a", fg="white").pack(side=tk.LEFT)
        tk.Label(filter_frame, text="%", bg="#3c3c3c", fg="white").pack(side=tk.LEFT)

        tk.Button(frame, text="Bandpass", command=self.apply_frequency_filter,
                 bg="#2a5a2a", fg="white").pack(fill=tk.X, padx=5, pady=1)
        tk.Button(frame, text="Bandstop", command=self.apply_frequency_filter_notch,
                 bg="#5a2a2a", fg="white").pack(fill=tk.X, padx=5, pady=1)

        # HPSS
        hpss_frame = tk.LabelFrame(container, text="🎵 Source Separation", bg="#3c3c3c", fg="cyan")
        hpss_frame.pack(fill=tk.X, pady=5)

        tk.Button(hpss_frame, text="Keep Harmonics Only", command=lambda: self.apply_hpss("harmonic"),
                 bg="#2a5a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)
        tk.Button(hpss_frame, text="Keep Percussive Only", command=lambda: self.apply_hpss("percussive"),
                 bg="#5a2a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)

        # Transform
        trans_frame = tk.LabelFrame(container, text="🔄 Geometric", bg="#3c3c3c", fg="cyan")
        trans_frame.pack(fill=tk.X, pady=5)

        tk.Button(trans_frame, text="↔️ Flip H", command=lambda: self.transform_image("horizontal"),
                 bg="#4c4c4c", fg="white").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        tk.Button(trans_frame, text="↕️ Flip V", command=lambda: self.transform_image("vertical"),
                 bg="#4c4c4c", fg="white").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)

    def setup_granular_tab(self):
        """Setup granular synthesis tab"""
        container = self.tab_contents["granular"]

        # Granular Controls
        frame = tk.LabelFrame(container, text="🎵 Granular Synthesis", bg="#3c3c3c", fg="cyan")
        frame.pack(fill=tk.X, pady=5)

        tk.Label(frame, text="Turn selections into playable grains!", bg="#3c3c3c", fg="yellow",
                font=("Arial", 8)).pack(pady=2)

        # Grain Size
        tk.Label(frame, text="Grain Size (samples)", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.grain_size_var = tk.IntVar(value=100)
        tk.Scale(frame, from_=20, to=500, orient=tk.HORIZONTAL, variable=self.grain_size_var,
                bg="#3c3c3c", fg="white", highlightthickness=0).pack(fill=tk.X, padx=5)

        # Grain Density
        tk.Label(frame, text="Density (grains/second)", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.grain_density_var = tk.IntVar(value=5)
        tk.Scale(frame, from_=1, to=20, orient=tk.HORIZONTAL, variable=self.grain_density_var,
                bg="#3c3c3c", fg="white", highlightthickness=0).pack(fill=tk.X, padx=5)

        # Grain Pitch
        tk.Label(frame, text="Pitch Variation", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.grain_pitch_var = tk.DoubleVar(value=1.0)
        tk.Scale(frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.grain_pitch_var,
                bg="#3c3c3c", fg="white", highlightthickness=0).pack(fill=tk.X, padx=5)

        # Grain Actions
        tk.Button(frame, text="🎵 Create Grains", command=self.create_grains,
                 bg="#2a5a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)
        tk.Button(frame, text="💫 Scatter Grains", command=self.scatter_grains,
                 bg="#5a5a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)
        tk.Button(frame, text="🌊 Grain Cloud", command=self.grain_cloud_texture,
                 bg="#2a5a5a", fg="white").pack(fill=tk.X, padx=5, pady=2)
        tk.Button(frame, text="🗑️ Clear Grains", command=self.clear_grains,
                 bg="#5a2a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)

        # Freeze
        freeze_frame = tk.LabelFrame(container, text="❄️ Spectral Freeze", bg="#3c3c3c", fg="cyan")
        freeze_frame.pack(fill=tk.X, pady=5)

        tk.Button(freeze_frame, text="❄️ Freeze Current", command=self.freeze_spectrum,
                 bg="#2a2a5a", fg="white").pack(fill=tk.X, padx=5, pady=2)
        tk.Button(freeze_frame, text="🌊 Smear Frozen", command=self.smear_spectrum,
                 bg="#5a2a5a", fg="white").pack(fill=tk.X, padx=5, pady=2)

    def setup_creative_tab(self):
        """Setup creative/unique features tab"""
        container = self.tab_contents["creative"]

        # Particle Spray
        particle_frame = tk.LabelFrame(container, text="💫 Particle Spray", bg="#3c3c3c", fg="cyan")
        particle_frame.pack(fill=tk.X, pady=5)

        tk.Label(particle_frame, text="Particle Intensity", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.particle_intensity = tk.IntVar(value=200)
        tk.Scale(particle_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.particle_intensity,
                bg="#3c3c3c", fg="white", highlightthickness=0).pack(fill=tk.X, padx=5)

        tk.Label(particle_frame, text="Spread", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.particle_spread_var = tk.IntVar(value=10)
        tk.Scale(particle_frame, from_=1, to=50, orient=tk.HORIZONTAL, variable=self.particle_spread_var,
                bg="#3c3c3c", fg="white", highlightthickness=0).pack(fill=tk.X, padx=5)

        # Noise Injection
        noise_frame = tk.LabelFrame(container, text="🔊 Noise Injection", bg="#3c3c3c", fg="cyan")
        noise_frame.pack(fill=tk.X, pady=5)

        tk.Label(noise_frame, text="Noise Amount", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.noise_amount = tk.IntVar(value=30)
        tk.Scale(noise_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.noise_amount,
                bg="#3c3c3c", fg="white", highlightthickness=0).pack(fill=tk.X, padx=5)

        tk.Button(noise_frame, text="⚪ White", command=lambda: self.inject_noise("white"),
                 bg="#4c4c4c", fg="white").pack(fill=tk.X, padx=5, pady=1)
        tk.Button(noise_frame, text="🔵 Blue", command=lambda: self.inject_noise("blue"),
                 bg="#2a4a5a", fg="white").pack(fill=tk.X, padx=5, pady=1)

        # Chaos/Glitch
        chaos_frame = tk.LabelFrame(container, text="🎲 Chaos & Glitch", bg="#3c3c3c", fg="cyan")
        chaos_frame.pack(fill=tk.X, pady=5)

        tk.Button(chaos_frame, text="🎲 Scatter", command=self.random_scatter,
                 bg="#5a2a5a", fg="white").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        tk.Button(chaos_frame, text="💢 Glitch", command=self.slice_and_glitch,
                 bg="#5a5a2a", fg="white").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)

    def setup_morph_tab(self):
        """Setup morphing tab"""
        container = self.tab_contents["morph"]

        # State Management
        state_frame = tk.LabelFrame(container, text="🔄 Morph States", bg="#3c3c3c", fg="cyan")
        state_frame.pack(fill=tk.X, pady=5)

        self.state_listbox = tk.Listbox(state_frame, bg="#2a2a2a", fg="white", height=5)
        self.state_listbox.pack(fill=tk.X, padx=5, pady=2)

        tk.Button(state_frame, text="💾 Save State", command=self.save_morph_state,
                 bg="#2a5a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)
        tk.Button(state_frame, text="🗑️ Delete State", command=self.delete_morph_state,
                 bg="#5a2a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)

        # Morph Controls
        tk.Label(state_frame, text="Morph Position", bg="#3c3c3c", fg="white").pack(anchor=tk.W)
        self.morph_slider = tk.Scale(state_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                    bg="#3c3c3c", fg="white", highlightthickness=0)
        self.morph_slider.set(0)
        self.morph_slider.pack(fill=tk.X, padx=5)

        tk.Button(state_frame, text="🔄 Preview", command=self.preview_morph,
                 bg="#5a5a2a", fg="white").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        tk.Button(state_frame, text="✅ Apply", command=self.apply_morph,
                 bg="#2a5a2a", fg="white").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)

        # Warp Markers
        warp_frame = tk.LabelFrame(container, text="⏯️ Warp Markers", bg="#3c3c3c", fg="cyan")
        warp_frame.pack(fill=tk.X, pady=5)

        tk.Button(warp_frame, text="Apply Warp", command=self.apply_warp,
                 bg="#2a5a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)
        tk.Button(warp_frame, text="Clear Markers", command=self.clear_warp_markers,
                 bg="#5a2a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)

    def setup_canvas_events(self):
        """Setup mouse, keyboard, and resize events"""
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-c>", lambda e: self.copy_selection())
        self.root.bind("<Control-v>", lambda e: self.paste_selection())
        self.root.bind("<Delete>", lambda e: self.delete_selection())
        self.root.bind("<Escape>", lambda e: self.clear_selection())

    # === Event Handlers ===

    def on_canvas_resize(self, event):
        """Handle dynamic canvas resizing with debounce for performance"""
        if abs(self.canvas_width - event.width) < 5 and abs(self.canvas_height - event.height) < 5:
            return
            
        self.canvas_width = max(10, event.width)
        self.canvas_height = max(10, event.height)
        
        # Cancel old timer
        if hasattr(self, '_resize_timer') and self._resize_timer:
            self.root.after_cancel(self._resize_timer)
            
        # Debounce to prevent lag during aggressive window resizing
        self._resize_timer = self.root.after(100, self._perform_resize_redraw)
        
    def _perform_resize_redraw(self):
        self._resize_timer = None
        if hasattr(self, 'time_ruler'):
            self.draw_time_ruler(self.time_ruler)
        if hasattr(self, 'freq_ruler'):
            self.draw_freq_ruler(self.freq_ruler)
            
        if hasattr(self, 'original_image') and self.original_image:
            self.update_canvas_display()
            
        # Update playhead line height
        coords = self.canvas.coords(self.playhead)
        if len(coords) == 4:
            self.canvas.coords(self.playhead, coords[0], 0, coords[2], self.canvas_height)

    def on_canvas_click(self, event):
        """Handle canvas click"""
        try:
            x, y = event.x, event.y

            if self.current_tool == Tool.WARP_MARKER:
                self.add_warp_marker(x, y)
                return

            if self.current_tool == Tool.SELECT:
                self.selection_start = (x, y)
                if self.selection_mode == SelectionMode.RECTANGLE:
                    self.canvas.coords(self.selection_rect, x, y, x, y)
                    self.canvas.itemconfigure(self.selection_rect, state="normal")

            elif self.current_tool in [Tool.BRUSH, Tool.ERASER]:
                self.drawing = True
                self.last_x = x
                self.last_y = y
                self.draw_point(x, y)

            elif self.current_tool == Tool.LINE:
                self.drawing = True
                self.selection_start = (x, y)

            elif self.current_tool == Tool.PARTICLE_SPRAY:
                self.spray_particles(x, y)

            elif self.current_tool == Tool.FILL:
                self.flood_fill(x, y)

            elif self.current_tool == Tool.GRANULAR:
                self.create_grains_at_point(x, y)

            elif self.current_tool == Tool.FREEZE:
                self.freeze_at_point(x, y)

            elif self.current_tool == Tool.SPECTRAL_SHAPER:
                self.add_spectral_point(x, y)

        except Exception as e:
            print(f"Error in on_canvas_click: {e}")

    def on_canvas_drag(self, event):
        """Handle canvas drag"""
        try:
            x, y = event.x, event.y

            if self.current_tool == Tool.SELECT and self.selection_start:
                if self.selection_mode == SelectionMode.RECTANGLE:
                    sx, sy = self.selection_start
                    self.canvas.coords(self.selection_rect, sx, sy, x, y)

            elif self.current_tool in [Tool.BRUSH, Tool.ERASER] and self.drawing:
                self.draw_line(self.last_x, self.last_y, x, y)
                self.last_x = x
                self.last_y = y

            elif self.current_tool == Tool.LINE and self.drawing:
                self.canvas.delete("line_preview")
                sx, sy = self.selection_start
                self.canvas.create_line(sx, sy, x, y, fill="white",
                                       width=self.brush_slider.get(), tags="line_preview")

            elif self.current_tool == Tool.PARTICLE_SPRAY:
                self.spray_particles(x, y)

        except Exception as e:
            print(f"Error in on_canvas_drag: {e}")

    def on_canvas_release(self, event):
        """Handle canvas release"""
        try:
            if self.current_tool == Tool.SELECT and self.selection_start:
                x, y = event.x, event.y
                sx, sy = self.selection_start

                x1, x2 = sorted([sx, x])
                y1, y2 = sorted([sy, y])

                if x2 - x1 > 2 and y2 - y1 > 2:
                    self.selection_coords = (x1, y1, x2, y2)
                    self.canvas.coords(self.selection_rect, x1, y1, x2, y2)
                    self.active_selection = "rectangle"
                    self.update_status(f"Selection: {x1},{y1} → {x2},{y2}")
                else:
                    self.clear_selection()

                self.selection_start = None

            elif self.current_tool == Tool.LINE and self.drawing:
                self.canvas.delete("line_preview")
                sx, sy = self.selection_start
                self.draw_line(sx, sy, event.x, event.y)
                self.drawing = False

            elif self.current_tool in [Tool.BRUSH, Tool.ERASER, Tool.PARTICLE_SPRAY]:
                self.drawing = False
                self.save_undo_state()

        except Exception as e:
            print(f"Error in on_canvas_release: {e}")

    def on_canvas_motion(self, event):
        pass

    # === Tool Methods ===

    def set_tool(self):
        self.current_tool = self.tool_var.get()
        self.update_status(f"Tool: {self.current_tool}")

    def set_selection_mode(self):
        self.selection_mode = self.selection_var.get()
        self.update_status(f"Selection: {self.selection_mode}")

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    # === Drawing Methods ===

    def draw_point(self, x, y):
        """Draw a single point with error checking"""
        try:
            if not self.modified_image or not self.original_image:
                return

            size = self.brush_slider.get()
            color = 255 if self.current_tool == Tool.BRUSH else 0

            # Prevent division by zero
            if self.canvas_width == 0 or self.canvas_height == 0:
                return

            draw = ImageDraw.Draw(self.modified_image)
            scale_x = self.original_image.width / self.canvas_width
            scale_y = self.original_image.height / self.canvas_height

            img_x = int(x * scale_x)
            img_y = int(y * scale_y)
            img_size = int(size * scale_x)

            draw.ellipse([img_x - img_size, img_y - img_size,
                         img_x + img_size, img_y + img_size], fill=color)
            self.update_canvas_display()
        except Exception as e:
            print(f"draw_point error: {e}")

    def draw_line(self, x1, y1, x2, y2):
        """Draw a line with error checking"""
        try:
            if not self.modified_image or not self.original_image:
                return

            size = self.brush_slider.get()
            color = 255 if self.current_tool == Tool.BRUSH else 0

            if self.canvas_width == 0 or self.canvas_height == 0:
                return

            draw = ImageDraw.Draw(self.modified_image)
            scale_x = self.original_image.width / self.canvas_width
            scale_y = self.original_image.height / self.canvas_height

            draw.line([int(x1 * scale_x), int(y1 * scale_y),
                      int(x2 * scale_x), int(y2 * scale_y)],
                     fill=color, width=int(size * scale_x))
            self.update_canvas_display()
        except Exception as e:
            print(f"draw_line error: {e}")

    def spray_particles(self, x, y):
        """Spray particles for granular effect with error checking"""
        try:
            if not self.modified_image or not self.original_image:
                return

            if not hasattr(self, 'particle_intensity'):
                return

            intensity = self.particle_intensity.get()
            spread = self.particle_spread_var.get()
            draw = ImageDraw.Draw(self.modified_image)

            if self.canvas_width == 0 or self.canvas_height == 0:
                return

            scale_x = self.original_image.width / self.canvas_width
            scale_y = self.original_image.height / self.canvas_height

            for _ in range(self.particle_density):
                dx = random.randint(-spread, spread)
                dy = random.randint(-spread, spread)
                px = int((x + dx) * scale_x)
                py = int((y + dy) * scale_y)
                size = random.randint(1, 3)

                if 0 <= px < self.modified_image.width and 0 <= py < self.modified_image.height:
                    pixel_val = random.randint(0, intensity)
                    draw.ellipse([px - size, py - size, px + size, py + size], fill=pixel_val)

            self.update_canvas_display()
        except Exception as e:
            print(f"spray_particles error: {e}")

    particle_density = 3

    def flood_fill(self, x, y):
        """Flood fill from point with error checking"""
        try:
            if not self.modified_image or not self.original_image:
                return

            if self.canvas_width == 0 or self.canvas_height == 0:
                return

            scale_x = self.original_image.width / self.canvas_width
            scale_y = self.original_image.height / self.canvas_height

            img_x = int(x * scale_x)
            img_y = int(y * scale_y)

            if not (0 <= img_x < self.modified_image.width and 0 <= img_y < self.modified_image.height):
                return

            pixels = self.modified_image.load()
            target_color = pixels[img_x, img_y]
            fill_color = 255

            if target_color == fill_color:
                return

            # Stack-based flood fill
            stack = [(img_x, img_y)]
            visited = set()

            while stack and len(visited) < 10000:
                px, py = stack.pop()

                if (px, py) in visited:
                    continue
                if not (0 <= px < self.modified_image.width and 0 <= py < self.modified_image.height):
                    continue
                if pixels[px, py] != target_color:
                    continue

                pixels[px, py] = fill_color
                visited.add((px, py))

                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    stack.append((px + dx, py + dy))

            self.update_canvas_display()
            self.save_undo_state()
        except Exception as e:
            print(f"flood_fill error: {e}")

    def update_canvas_display(self):
        """Update canvas with current image, preserving playhead position"""
        if self.modified_image and self.canvas_width > 0 and self.canvas_height > 0:
            display_img = self.modified_image.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(display_img)
            self.canvas.delete("img_bg")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="img_bg")
            self.canvas.tag_raise(self.playhead)

    def save_undo_state(self):
        if self.modified_image:
            self.undo_stack.append(self.modified_image.copy())
            if len(self.undo_stack) > 20:
                self.undo_stack.pop(0)
            self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.modified_image.copy())
            self.modified_image = self.undo_stack.pop()
            self.original_image = self.modified_image.copy()
            self.update_canvas_display()
            self.update_status("↶ Undo")
        else:
            self.update_status("Nothing to undo")

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.modified_image.copy())
            self.modified_image = self.redo_stack.pop()
            self.original_image = self.modified_image.copy()
            self.update_canvas_display()
            self.update_status("↷ Redo")
        else:
            self.update_status("Nothing to redo")

    # === Selection Operations ===

    def clear_selection(self):
        self.canvas.itemconfigure(self.selection_rect, state="hidden")
        self.selection_coords = None
        self.active_selection = None
        self.update_status("Selection cleared")

    def copy_selection(self):
        if not self.modified_image or not self.selection_coords:
            self.update_status("No selection")
            return

        x1, y1, x2, y2 = self.selection_coords
        scale_x = self.modified_image.width / self.canvas_width
        scale_y = self.modified_image.height / self.canvas_height

        ix1 = int(x1 * scale_x)
        iy1 = int(y1 * scale_y)
        ix2 = int(x2 * scale_x)
        iy2 = int(y2 * scale_y)

        self.clipboard = self.modified_image.crop((ix1, iy1, ix2, iy2))
        self.update_status("Copied to clipboard")

    def paste_selection(self):
        if not self.clipboard:
            self.update_status("Nothing to paste")
            return

        self.save_undo_state()
        x = (self.modified_image.width - self.clipboard.width) // 2
        y = (self.modified_image.height - self.clipboard.height) // 2
        self.modified_image.paste(self.clipboard, (x, y))
        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        self.update_status("Pasted")

    def delete_selection(self):
        if not self.modified_image or not self.selection_coords:
            self.update_status("No selection")
            return

        self.save_undo_state()
        x1, y1, x2, y2 = self.selection_coords
        scale_x = self.modified_image.width / self.canvas_width
        scale_y = self.modified_image.height / self.canvas_height

        draw = ImageDraw.Draw(self.modified_image)
        draw.rectangle([int(x1 * scale_x), int(y1 * scale_y),
                       int(x2 * scale_x), int(y2 * scale_y)], fill=0)

        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        self.update_status("Deleted selection")

    # === Effects ===

    def apply_live_effects(self, event=None):
        if not self.original_image:
            return

        img = self.original_image.copy()

        # Brightness
        b = self.brightness_slider.get()
        if b != 0:
            img = Image.eval(img, lambda x: max(0, min(255, x + b)))

        # Contrast
        c = self.contrast_slider.get()
        if c != 1.0:
            img = Image.eval(img, lambda x: max(0, min(255, (x - 128) * c + 128)))

        # Gamma
        g = self.gamma_slider.get()
        if g != 1.0:
            img = Image.eval(img, lambda x: max(0, min(255, 255 * (x / 255) ** (1 / g))))

        # Tint
        if self.tint_color:
            r, g, b = self.tint_color
            tint = Image.new('RGB', img.size, (r, g, b))
            img_rgb = img.convert('RGB')
            mask = img
            img_rgb.paste(tint, (0, 0), mask)
            img = img_rgb.convert('L')

        # Invert
        if self.invert_var.get():
            img = Image.eval(img, lambda x: 255 - x)

        if self.canvas_width > 0 and self.canvas_height > 0:
            display_img = img.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(display_img)
            self.canvas.delete("img_bg")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="img_bg")
            self.canvas.tag_raise(self.playhead)

    def apply_effects_permanent(self):
        if not self.original_image:
            return

        self.save_undo_state()
        self.apply_live_effects()

        display_img = ImageTk.getimage(self.tk_image)
        if display_img.mode != 'L':
            display_img = display_img.convert('L')
        self.modified_image = display_img.resize((self.original_image.width, self.original_image.height), Image.Resampling.LANCZOS)
        self.original_image = self.modified_image.copy()

        self.reset_effects()
        self.update_canvas_display()
        self.update_status("Effects applied")

    def reset_effects(self):
        self.brightness_slider.set(0)
        self.contrast_slider.set(1.0)
        self.gamma_slider.set(1.0)
        self.invert_var.set(False)
        self.tint_color = None
        self.tint_button.config(text="No tint", bg="gray")
        self.update_canvas_display()

    def apply_all_and_regenerate(self):
        if not self.original_image:
            self.update_status("No image loaded")
            return

        self.update_status("Applying effects and regenerating audio...")
        self.apply_effects_permanent()

        if hasattr(self, 'modified_image') and self.modified_image:
            if AUDIO_AVAILABLE:
                self.stop_audio(reset_needle=True)
                threading.Thread(target=self._regenerate_audio_from_image, args=(self.modified_image,), daemon=True).start()
            else:
                self.update_status("Audio not available")
        else:
            self.update_status("No modified image to regenerate from")

    def _regenerate_audio_from_image(self, image):
        try:
            self.root.after(0, lambda: self.update_status("Regenerating audio..."))

            target_height = (self.n_fft // 2) + 1
            img = image.resize((image.width, target_height), Image.Resampling.LANCZOS)

            img_data = np.array(img)
            img_data = np.flipud(img_data)

            db_data = (img_data / 255.0) * 80.0 - 80.0
            amplitude_matrix = librosa.db_to_amplitude(db_data)

            audio_data = librosa.griffinlim(amplitude_matrix, n_iter=32, hop_length=self.hop_length)
            audio_data = audio_data / np.max(np.abs(audio_data))

            self.current_audio = audio_data
            if self.current_audio is not None:
                self.audio_duration = len(self.current_audio) / self.sample_rate
                with self.audio_lock:
                    self.audio_index = 0.0
                self.root.after(0, lambda: self.update_status(f"Ready! Duration: {self.audio_duration:.2f}s"))
                self.root.after(0, lambda: self.canvas.coords(self.playhead, 0, 0, 0, self.canvas_height))
                self.root.after(0, lambda: self.draw_time_ruler(self.time_ruler))
                self.root.after(0, lambda: self.canvas.itemconfigure(self.playhead, state="normal"))
            else:
                self.root.after(0, lambda: self.update_status("Audio regeneration failed"))
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"Audio error: {str(e)}"))

    def choose_tint(self):
        color = colorchooser.askcolor(title="Choose Tint Color")
        if color[0]:
            self.tint_color = tuple(map(int, color[0]))
            self.tint_button.config(text=f"RGB: {self.tint_color}", bg=color[1])
            self.apply_live_effects()

    # === Spectral Processing ===

    def apply_vertical_scale(self):
        if not self.modified_image:
            return

        self.save_undo_state()
        scale = self.vertical_scale_slider.get()
        if scale == 1.0:
            return

        width, height = self.modified_image.size
        new_height = int(height * scale)

        if scale > 1.0:
            scaled = self.modified_image.resize((width, new_height), Image.Resampling.LANCZOS)
            result = Image.new('L', (width, height))
            y_offset = max(0, (new_height - height) // 2)
            result.paste(scaled.crop((0, y_offset, width, y_offset + height)))
            self.modified_image = result
        else:
            scaled = self.modified_image.resize((width, new_height), Image.Resampling.LANCZOS)
            result = Image.new('L', (width, height))
            y_offset = (height - new_height) // 2
            result.paste(scaled, (0, y_offset))
            self.modified_image = result

        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        self.update_status(f"Vertical scale {scale}x")

    def apply_frequency_shift(self):
        if not self.modified_image:
            return

        self.save_undo_state()
        shift_hz = self.freq_shift_slider.get()
        if shift_hz == 0:
            return

        max_freq = self.sample_rate / 2
        shift_pct = shift_hz / max_freq
        shift_pixels = int(shift_pct * self.modified_image.height)

        if shift_pixels != 0:
            if shift_pixels > 0:
                self.modified_image = Image.fromarray(np.roll(np.array(self.modified_image), shift_pixels, axis=0), mode='L')
            else:
                self.modified_image = Image.fromarray(np.roll(np.array(self.modified_image), shift_pixels, axis=0), mode='L')

        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        self.update_status(f"Frequency shifted {shift_hz} Hz")

    def apply_frequency_filter(self):
        self._apply_frequency_filter(True)

    def apply_frequency_filter_notch(self):
        self._apply_frequency_filter(False)

    def _apply_frequency_filter(self, pass_band):
        if not self.modified_image:
            return

        self.save_undo_state()
        low = self.filter_low.get()
        high = self.filter_high.get()

        width, height = self.modified_image.size
        pixels = self.modified_image.load()

        y_low = int(height * (1.0 - high / 100.0))
        y_high = int(height * (1.0 - low / 100.0))

        for y in range(height):
            in_band = y_low <= y <= y_high
            if (pass_band and not in_band) or (not pass_band and in_band):
                for x in range(width):
                    pixels[x, y] = 0

        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        mode = "Bandpass" if pass_band else "Bandstop"
        self.update_status(f"{mode}: {low}%-{high}%")

    def apply_hpss(self, keep):
        if not AUDIO_AVAILABLE:
            self.update_status("HPSS requires audio libraries")
            return

        self.update_status("HPSS processing...")
        self.save_undo_state()

        img_array = np.array(self.modified_image).astype(float)
        from scipy.ndimage import gaussian_filter

        if keep == "harmonic":
            smoothed = gaussian_filter(img_array, sigma=(0, 5))
            result = np.where(smoothed > 50, img_array, 0)
        else:
            smoothed = gaussian_filter(img_array, sigma=(5, 0))
            result = np.where(smoothed > 50, img_array, 0)

        self.modified_image = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode='L')
        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        self.update_status(f"Kept {keep}")

    def transform_image(self, transform):
        if not self.modified_image:
            return

        self.save_undo_state()

        if transform == "horizontal":
            self.modified_image = self.modified_image.transpose(Image.FLIP_LEFT_RIGHT)
        elif transform == "vertical":
            self.modified_image = self.modified_image.transpose(Image.FLIP_TOP_BOTTOM)

        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        self.update_status(f"Applied {transform}")

    # === Granular Synthesis ===

    def create_grains(self):
        if not self.selection_coords:
            self.update_status("Select a region first")
            return

        self.grains.clear()
        x1, y1, x2, y2 = self.selection_coords
        grain_size = self.grain_size_var.get()
        density = self.grain_density_var.get()
        pitch = self.grain_pitch_var.get()

        for i in range(density * 10):
            gx = random.randint(x1, x2)
            gy = random.randint(y1, y2)
            self.grains.append({
                'x': gx, 'y': gy, 'size': grain_size,
                'pitch': pitch, 'intensity': random.randint(100, 255)
            })

        self.update_status(f"Created {len(self.grains)} grains")

    def create_grains_at_point(self, x, y):
        try:
            if not hasattr(self, 'grain_density_var'):
                return

            for _ in range(self.grain_density_var.get()):
                self.grains.append({
                    'x': x + random.randint(-20, 20),
                    'y': y + random.randint(-20, 20),
                    'size': self.grain_size_var.get(),
                    'pitch': self.grain_pitch_var.get(),
                    'intensity': random.randint(100, 255)
                })

            if self.modified_image and self.canvas_width > 0 and self.canvas_height > 0:
                scale_x = self.modified_image.width / self.canvas_width
                scale_y = self.modified_image.height / self.canvas_height
                draw = ImageDraw.Draw(self.modified_image)

                for grain in self.grains[-20:]:
                    gx = int(grain['x'] * scale_x)
                    gy = int(grain['y'] * scale_y)
                    size = int(grain['size'] * scale_x / 10)
                    draw.ellipse([gx - size, gy - size, gx + size, gy + size],
                                fill=grain['intensity'])

                self.update_canvas_display()

            self.update_status(f"Added grains at ({x}, {y})")
        except Exception as e:
            print(f"create_grains_at_point error: {e}")

    def scatter_grains(self):
        if not self.grains:
            self.update_status("No grains to scatter")
            return

        self.save_undo_state()

        if self.modified_image:
            scale_x = self.modified_image.width / self.canvas_width
            scale_y = self.modified_image.height / self.canvas_height
            draw = ImageDraw.Draw(self.modified_image)

            for grain in self.grains:
                gx = int(grain['x'] * scale_x + random.randint(-50, 50))
                gy = int(grain['y'] * scale_y + random.randint(-50, 50))
                size = int(grain['size'] * scale_x / 10)
                intensity = grain['intensity']

                if 0 <= gx < self.modified_image.width and 0 <= gy < self.modified_image.height:
                    draw.ellipse([gx - size, gy - size, gx + size, gy + size], fill=intensity)

            self.original_image = self.modified_image.copy()
            self.update_canvas_display()
            self.update_status("Grains scattered")

    def grain_cloud_texture(self):
        if not self.grains:
            self.update_status("No grains")
            return

        self.save_undo_state()

        if self.modified_image:
            scale_x = self.modified_image.width / self.canvas_width
            scale_y = self.modified_image.height / self.canvas_height

            cloud = Image.new('L', self.modified_image.size, 0)
            draw = ImageDraw.Draw(cloud)

            for grain in self.grains:
                for _ in range(5):
                    gx = int(grain['x'] * scale_x + random.gauss(0, 30))
                    gy = int(grain['y'] * scale_y + random.gauss(0, 30))
                    size = int(grain['size'] * scale_x / 8)
                    intensity = int(grain['intensity'] * random.uniform(0.5, 1.0))

                    if 0 <= gx < self.modified_image.width and 0 <= gy < self.modified_image.height:
                        draw.ellipse([gx - size, gy - size, gx + size, gy + size], fill=intensity)

            self.modified_image = Image.blend(self.modified_image, cloud, 0.5)
            self.original_image = self.modified_image.copy()
            self.update_canvas_display()
            self.update_status("Grain cloud created")

    def clear_grains(self):
        self.grains.clear()
        self.update_status("Grains cleared")

    # === Spectral Freeze ===

    def freeze_spectrum(self):
        if self.modified_image:
            self.freeze_buffer = self.modified_image.copy()
            self.freeze_active = True
            self.update_status("Spectrum frozen")

    def freeze_at_point(self, x, y):
        try:
            if not self.modified_image or self.canvas_width == 0:
                return

            if not self.freeze_buffer:
                self.freeze_buffer = self.modified_image.copy()

            scale_x = self.modified_image.width / self.canvas_width
            col = int(x * scale_x)

            if 0 <= col < self.modified_image.width:
                column_data = [self.modified_image.getpixel((col, r))
                              for r in range(self.modified_image.height)]
                self.freeze_buffer = Image.new('L', self.modified_image.size, 0)
                draw = ImageDraw.Draw(self.freeze_buffer)

                for c in range(self.modified_image.width):
                    for r in range(self.modified_image.height):
                        draw.point((c, r), fill=column_data[r])

            self.freeze_active = True
            self.update_status(f"Frozen column at {x}")
        except Exception as e:
            print(f"freeze_at_point error: {e}")

    def smear_spectrum(self):
        if not self.freeze_buffer:
            self.update_status("No frozen spectrum")
            return

        self.save_undo_state()

        if self.modified_image:
            self.modified_image = Image.blend(self.modified_image, self.freeze_buffer, 0.3)
            self.original_image = self.modified_image.copy()
            self.update_canvas_display()
            self.update_status("Spectrum smeared")

    # === Creative Effects ===

    def inject_noise(self, noise_type):
        if not self.modified_image:
            return

        self.save_undo_state()

        if self.modified_image.mode != 'L':
            self.modified_image = self.modified_image.convert('L')

        width, height = self.modified_image.size
        amount = self.noise_amount.get()

        if noise_type == "white":
            noise = np.random.randint(0, amount, (height, width))
        else:  # blue
            noise = np.random.randn(height, width)
            noise = np.diff(noise, axis=0, prepend=0)
            noise = np.abs(noise)
            noise = noise / noise.max() * amount

        img_array = np.array(self.modified_image).astype(float)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        self.modified_image = Image.fromarray(img_array, mode='L')
        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        self.update_status(f"{noise_type.capitalize()} noise injected")

    def random_scatter(self):
        if not self.modified_image:
            return

        self.save_undo_state()
        img_array = np.array(self.modified_image)
        height, width = img_array.shape

        for _ in range(100):
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            x2 = random.randint(0, width - 1)
            y2 = random.randint(0, height - 1)
            block_size = random.randint(5, 20)

            if (x1 + block_size < width and y1 + block_size < height and
                x2 + block_size < width and y2 + block_size < height):
                block1 = img_array[y1:y1+block_size, x1:x1+block_size].copy()
                block2 = img_array[y2:y2+block_size, x2:x2+block_size].copy()
                img_array[y1:y1+block_size, x1:x1+block_size] = block2
                img_array[y2:y2+block_size, x2:x2+block_size] = block1

        self.modified_image = Image.fromarray(img_array, mode="L")
        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        self.update_status("Random scatter applied")

    def slice_and_glitch(self):
        if not self.modified_image:
            return

        self.save_undo_state()
        img_array = np.array(self.modified_image)
        height, width = img_array.shape

        num_slices = random.randint(5, 15)
        slice_height = height // num_slices

        for i in range(num_slices):
            y_start = i * slice_height
            y_end = min((i + 1) * slice_height, height)
            offset = random.randint(-50, 50)
            slice_data = img_array[y_start:y_end, :].copy()
            img_array[y_start:y_end, :] = np.roll(slice_data, offset, axis=1)

            if random.random() > 0.5:
                img_array[y_start:y_end, :] = np.clip(
                    img_array[y_start:y_end, :] * random.uniform(0.5, 1.5), 0, 255
                )

        self.modified_image = Image.fromarray(img_array.astype(np.uint8), mode='L')
        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        self.update_status("Glitch applied")

    def add_spectral_point(self, x, y):
        try:
            if self.canvas_height == 0:
                return

            scale_y = self.canvas_height / 255
            display_y = int(y / scale_y)
            self.spectral_curve.append((x, display_y))

            self.canvas.delete("shaper_curve")
            if len(self.spectral_curve) > 1:
                points = []
                for px, py in self.spectral_curve:
                    points.extend([px, py])
                self.canvas.create_line(points, fill="cyan", width=2, tags="shaper_curve", smooth=True)
            self.update_status(f"Spectral point added")
        except Exception as e:
            print(f"add_spectral_point error: {e}")

    # === Warp Markers ===

    def add_warp_marker(self, x, y):
        stretch = 1.0 + (self.canvas_height - y) / max(1, self.canvas_height)
        self.warp_markers.append((x, stretch))

        line = self.canvas.create_line(x, 0, x, self.canvas_height,
                                      fill="magenta", width=1, dash=(3, 3))
        self.warp_lines.append(line)
        self.update_status(f"Warp marker at x={x}, stretch={stretch:.2f}")

    def clear_warp_markers(self):
        self.warp_markers.clear()
        for line in self.warp_lines:
            self.canvas.delete(line)
        self.warp_lines.clear()
        self.update_status("Warp markers cleared")

    def apply_warp(self):
        if not self.warp_markers or not self.modified_image:
            self.update_status("No warp markers")
            return

        self.save_undo_state()
        self.warp_markers.sort(key=lambda m: m[0])
        img_array = np.array(self.modified_image).copy()
        width, height = self.modified_image.size

        prev_x = 0
        for x, stretch in self.warp_markers:
            img_x = int(x * width / self.canvas_width)
            if img_x > prev_x:
                segment = img_array[:, prev_x:img_x]
                new_height = int(height * stretch)
                if new_height != height:
                    if stretch > 1.0:
                        stretched = Image.fromarray(segment).resize((img_x - prev_x, new_height), Image.Resampling.LANCZOS)
                        y_offset = (new_height - height) // 2
                        stretched_array = np.array(stretched)[y_offset:y_offset+height, :]
                    else:
                        stretched = Image.fromarray(segment).resize((img_x - prev_x, new_height), Image.Resampling.LANCZOS)
                        y_offset = (height - new_height) // 2
                        temp = np.zeros((height, img_x - prev_x), dtype=np.uint8)
                        temp[y_offset:y_offset+new_height, :] = np.array(stretched)
                        stretched_array = temp
                    img_array[:, prev_x:img_x] = stretched_array
            prev_x = img_x

        self.modified_image = Image.fromarray(img_array, mode="L")
        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        self.update_status("Warp applied")

    # === Morphing ===

    def save_morph_state(self):
        if not self.modified_image:
            self.update_status("No image to save")
            return

        name = f"State {len(self.morph_states) + 1}"
        state = MorphState(self.modified_image, name)
        self.morph_states.append(state)
        self.state_listbox.insert(tk.END, name)
        self.update_status(f"Saved {name}")

    def delete_morph_state(self):
        selection = self.state_listbox.curselection()
        if selection:
            idx = selection[0]
            self.morph_states.pop(idx)
            self.state_listbox.delete(idx)
            self.update_status("State deleted")

    def preview_morph(self):
        if len(self.morph_states) < 2:
            self.update_status("Need at least 2 states")
            return

        morph_pos = self.morph_slider.get() / 100.0
        state1 = self.morph_states[0].image
        state2 = self.morph_states[1].image

        arr1 = np.array(state1).astype(float)
        arr2 = np.array(state2).astype(float)

        morphed = arr1 * (1 - morph_pos) + arr2 * morph_pos
        morphed = np.clip(morphed, 0, 255).astype(np.uint8)

        morphed_img = Image.fromarray(morphed)
        display_img = morphed_img.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(display_img)
        self.canvas.delete("img_bg")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="img_bg")
        self.update_status(f"Preview morph at {morph_pos:.0%}")

    def apply_morph(self):
        if len(self.morph_states) < 2:
            self.update_status("Need at least 2 states")
            return

        self.save_undo_state()
        morph_pos = self.morph_slider.get() / 100.0
        state1 = self.morph_states[0].image
        state2 = self.morph_states[1].image

        arr1 = np.array(state1).astype(float)
        arr2 = np.array(state2).astype(float)

        morphed = arr1 * (1 - morph_pos) + arr2 * morph_pos
        morphed = np.clip(morphed, 0, 255).astype(np.uint8)

        self.modified_image = Image.fromarray(morphed, mode="L")
        self.original_image = self.modified_image.copy()
        self.update_canvas_display()
        self.update_status(f"Morph applied at {morph_pos:.0%}")

    # === File Operations ===

    def load_image_and_process(self):
        img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not img_path:
            return

        self.stop_audio()
        self.update_status("Loading...")

        try:
            self.original_image = Image.open(img_path).convert('L')
            self.modified_image = self.original_image.copy()

            display_img = self.original_image.resize((max(10, self.canvas_width), max(10, self.canvas_height)), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(display_img)
            self.canvas.delete("img_bg")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="img_bg")
            self.canvas.tag_raise(self.playhead)

            self.img_path_for_export = img_path
            self.update_status("Image loaded")

            if AUDIO_AVAILABLE:
                threading.Thread(target=self._generate_audio_thread, args=(img_path,), daemon=True).start()
            else:
                self.update_status("Image loaded (Audio not available)")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {str(e)}")

    def _generate_audio_thread(self, img_path):
        try:
            self.current_audio = self.process_image_to_audio(img_path)
            if self.current_audio is not None:
                self.audio_duration = len(self.current_audio) / self.sample_rate
                with self.audio_lock:
                    self.audio_index = 0.0
                self.root.after(0, lambda: self.update_status(f"Ready! Duration: {self.audio_duration:.2f}s"))
                self.root.after(0, lambda: self.canvas.coords(self.playhead, 0, 0, 0, self.canvas_height))
                self.root.after(0, lambda: self.canvas.itemconfigure(self.playhead, state="normal"))
                self.root.after(0, lambda: self.draw_time_ruler(self.time_ruler))
            else:
                self.root.after(0, lambda: self.update_status("Audio processing failed"))
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"Audio error: {str(e)}"))

    def process_image_to_audio(self, img_path):
        try:
            img = Image.open(img_path).convert('L')
            target_height = (self.n_fft // 2) + 1
            img = img.resize((img.width, target_height), Image.Resampling.LANCZOS)

            img_data = np.array(img)
            img_data = np.flipud(img_data)

            db_data = (img_data / 255.0) * 80.0 - 80.0
            amplitude_matrix = librosa.db_to_amplitude(db_data)

            audio_data = librosa.griffinlim(amplitude_matrix, n_iter=32, hop_length=self.hop_length)
            audio_data = audio_data / np.max(np.abs(audio_data))
            return audio_data
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None

    def save_image(self):
        if not self.modified_image:
            messagebox.showwarning("Warning", "No image to save")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if save_path:
            try:
                self.modified_image.save(save_path)
                self.update_status(f"Saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed: {str(e)}")

    def image_to_wav(self):
        if not AUDIO_AVAILABLE:
            messagebox.showwarning("Warning", "Audio not available")
            return
        if not self.modified_image:
            messagebox.showwarning("Warning", "No image")
            return

        self.update_status("Generating audio...")
        temp_path = "sounder_export_temp.png"
        self.modified_image.save(temp_path)
        audio = self.process_image_to_audio(temp_path)

        if audio is None:
            messagebox.showerror("Error", "Failed to generate audio")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if save_path:
            try:
                sf.write(save_path, audio, self.sample_rate)
                self.update_status(f"Saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed: {str(e)}")

    def wav_to_image(self):
        if not AUDIO_AVAILABLE:
            messagebox.showwarning("Warning", "Audio not available")
            return

        wav_path = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if not wav_path:
            return

        try:
            self.update_status("Processing...")
            y, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
            D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img_data = np.clip((S_db + 80.0) / 80.0 * 255.0, 0, 255)
            img_data = np.flipud(img_data).astype(np.uint8)
            img = Image.fromarray(img_data, mode='L')

            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            if save_path:
                img.save(save_path)
                self.update_status(f"Saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")

    # === Playback ===

    def play_audio(self):
        if not AUDIO_AVAILABLE:
            messagebox.showwarning("Warning", "Audio not available")
            return
        if self.current_audio is None:
            messagebox.showwarning("Warning", "Load image and wait for processing")
            return

        if self.is_playing:
            self.stop_audio(reset_needle=False)

        audio_len = len(self.current_audio)
        with self.audio_lock:
            if self.audio_index >= audio_len or self.audio_index < 0:
                self.audio_index = 0.0
            current_idx = self.audio_index

        try:
            self.is_playing = True
            self.canvas.itemconfigure(self.playhead, state="normal")
            current_x = int((current_idx / audio_len) * self.canvas_width)
            self.canvas.coords(self.playhead, current_x, 0, current_x, self.canvas_height)
            self.canvas.tag_raise(self.playhead)

            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._live_audio_callback,
                blocksize=1024
            )
            self.stream.start()
            self.animate_needle()
            self.update_status("Playing...")

        except Exception as e:
            self.is_playing = False
            self.canvas.itemconfigure(self.playhead, state="hidden")
            messagebox.showerror("Error", f"Playback failed: {str(e)}")

    def _live_audio_callback(self, outdata, frames, time_info, status):
        try:
            if not self.is_playing or self.current_audio is None:
                outdata.fill(0)
                raise sd.CallbackStop()

            try:
                val = float(self.speed_slider.get())
            except:
                val = 1.0

            if val <= -1.0:
                speed = 1.0 / abs(val)
            elif val >= 1.0:
                speed = val
            elif val > 0.0:
                speed = max(0.1, val)
            elif val < 0.0:
                speed = -max(0.1, abs(val))
            else:
                speed = 1.0

            audio_len = len(self.current_audio)
            reached_end = False

            with self.audio_lock:
                current_idx = float(self.audio_index) if isinstance(self.audio_index, (int, float)) else 0.0
                for i in range(frames):
                    pos = current_idx + i * speed
                    if pos < 0 or pos >= audio_len:
                        outdata[i, 0] = 0.0
                        if pos >= audio_len:
                            reached_end = True
                    else:
                        idx = int(pos)
                        frac = pos - idx
                        if idx + 1 < audio_len:
                            outdata[i, 0] = self.current_audio[idx] * (1 - frac) + self.current_audio[idx + 1] * frac
                        else:
                            outdata[i, 0] = self.current_audio[idx]
                self.audio_index = current_idx + frames * speed

            if reached_end:
                self.root.after(0, self._on_playback_complete)
                raise sd.CallbackStop()

        except sd.CallbackStop:
            raise
        except Exception as e:
            outdata.fill(0)

    def _on_playback_complete(self):
        self.is_playing = False
        self.update_status("Playback complete")

    def animate_needle(self):
        if not self.is_playing:
            return
        if self.current_audio is None or len(self.current_audio) == 0:
            return

        if self.is_playing:
            self.root.after(30, self.animate_needle)

        try:
            with self.audio_lock:
                try:
                    audio_index = float(self.audio_index)
                except (TypeError, ValueError):
                    self.audio_index = 0.0
                    audio_index = 0.0

            audio_length = len(self.current_audio)
            if audio_length == 0: return

            audio_index = max(0.0, min(audio_index, float(audio_length)))
            progress = audio_index / audio_length
            progress = max(0.0, min(1.0, progress))

            x = int(progress * self.canvas_width)

            self.canvas.coords(self.playhead, x, 0, x, self.canvas_height)
            self.canvas.itemconfigure(self.playhead, state="normal")
            self.canvas.tag_raise(self.playhead)
        except Exception as e:
            pass

    def stop_audio(self, reset_needle=True):
        self.is_playing = False
        if self.stream:
            try:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
            except Exception as e:
                pass
            self.stream = None

        if reset_needle:
            with self.audio_lock:
                self.audio_index = 0.0
            self.canvas.coords(self.playhead, 0, 0, 0, self.canvas_height)
            self.canvas.itemconfigure(self.playhead, state="hidden")

    def seek_start(self):
        with self.audio_lock:
            self.audio_index = 0.0
        self.canvas.coords(self.playhead, 0, 0, 0, self.canvas_height)

    def open_visual_spectrogram(self):
        if not self.modified_image:
            messagebox.showwarning("Warning", "Load an image first")
            return
        VisualSpectrogramWindow(self.root, self.modified_image)


class VisualSpectrogramWindow:
    """
    SPECTROSCOPE SIMULATOR - Simulates real spectroscopic analysis of burning gas/light sources
    """
    ELEMENTAL_LINES = {
        'Hydrogen': [656.3, 486.1, 434.0, 410.2],
        'Helium': [587.6, 447.1, 501.6, 667.8, 706.5],
        'Sodium': [589.0, 589.6],
        'Mercury': [435.8, 546.1, 578.0, 404.7],
        'Neon': [585.2, 640.2, 703.2, 650.6],
        'Argon': [696.5, 706.7, 727.3, 738.4],
        'Carbon': [426.7, 564.9, 658.8],
        'Nitrogen': [648.2, 500.5, 444.7],
        'Oxygen': [557.7, 630.0, 636.4],
        'Iron': [525.0, 526.1, 527.0],
        'Fraunhofer': {
            'A': 759.4, 'B': 686.7, 'C': 656.3, 'D1': 589.6, 'D2': 589.0,
            'E': 527.0, 'F': 486.1, 'G': 434.0, 'H': 396.8
        }
    }

    def wavelength_to_rgb(self, wavelength):
        wavelength = float(wavelength)
        if 380 <= wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** 0.8
            G = 0.0
            B = (1.0 * attenuation) ** 0.8
        elif 440 <= wavelength <= 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440)) ** 0.8
            B = 1.0
        elif 490 <= wavelength <= 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490)) ** 0.8
        elif 510 <= wavelength <= 580:
            R = ((wavelength - 510) / (580 - 510)) ** 0.8
            G = 1.0
            B = 0.0
        elif 580 <= wavelength <= 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580)) ** 0.8
            B = 0.0
        elif 645 <= wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation) ** 0.8
            G = 0.0
            B = 0.0
        else:
            return (0, 0, 0)
        return (int(R * 255), int(G * 255), int(B * 255))

    def __init__(self, parent, source_image):
        self.source_image = source_image
        self.window = tk.Toplevel(parent)
        self.window.title("🔬 SPECTROSCOPE SIMULATOR - Burning Gas Analysis")
        
        # Scale spectrogram window dynamically
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        spec_w = min(1200, int(screen_width * 0.7))
        spec_h = min(800, int(screen_height * 0.7))
        
        # Center the window and shift slightly up
        x = int((screen_width - spec_w) / 2)
        y = max(0, int((screen_height - spec_h) / 2) - 40)
        
        self.window.geometry(f"{spec_w}x{spec_h}+{x}+{y}")
        self.window.minsize(800, 500)
        self.window.configure(bg="#0a0a0a")

        self.ribbon_width = 100
        self.ribbon_count = 8
        self.wavelength_range = (380, 750)
        self.dispersion_power = 1.0
        self.line_intensity_threshold = 50
        self.absorption_mode = False
        self.show_element_labels = True
        self.background_continuum = True
        self.noise_level = 5
        self.detected_elements = []

        self.setup_ui()
        self.analyze_spectrum()

    def setup_ui(self):
        header = tk.Frame(self.window, bg="#0a0a0a")
        header.pack(fill=tk.X, pady=5)

        title = tk.Label(header, text="🔬 OPTICAL SPECTROSCOPE SIMULATOR",
                        font=("Courier New", 16, "bold"), bg="#0a0a0a", fg="#00ff00")
        title.pack()

        subtitle = tk.Label(header,
                           text="Analyzing spectral emissions as from burning gas/stellar sources",
                           font=("Courier New", 10), bg="#0a0a0a", fg="#888888")
        subtitle.pack()

        main_content = tk.Frame(self.window, bg="#0a0a0a")
        main_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        control_panel = tk.Frame(main_content, bg="#1a1a1a", width=280, relief=tk.RIDGE, bd=2)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_panel.pack_propagate(False)

        tk.Label(control_panel, text="⚙️ SPECTROMETER SETTINGS",
                font=("Courier New", 11, "bold"), bg="#1a1a1a", fg="#00ff00").pack(pady=(10, 5))

        tk.Label(control_panel, text="Slit Width (samples):",
                bg="#1a1a1a", fg="white").pack(anchor=tk.W, padx=5)
        self.slit_width_slider = tk.Scale(control_panel, from_=20, to=500,
                                          orient=tk.HORIZONTAL, bg="#1a1a1a",
                                          fg="white", highlightthickness=0,
                                          command=lambda v: self.analyze_spectrum())
        self.slit_width_slider.set(100)
        self.slit_width_slider.pack(fill=tk.X, padx=5)

        tk.Label(control_panel, text="Exposure Stacks:",
                bg="#1a1a1a", fg="white").pack(anchor=tk.W, padx=5)
        self.exposure_slider = tk.Scale(control_panel, from_=1, to=20,
                                       orient=tk.HORIZONTAL, bg="#1a1a1a",
                                       fg="white", highlightthickness=0,
                                       command=lambda v: self.analyze_spectrum())
        self.exposure_slider.set(8)
        self.exposure_slider.pack(fill=tk.X, padx=5)

        tk.Label(control_panel, text="Prism Dispersion:",
                bg="#1a1a1a", fg="white").pack(anchor=tk.W, padx=5)
        self.dispersion_slider = tk.Scale(control_panel, from_=0.5, to=3.0,
                                         resolution=0.1, orient=tk.HORIZONTAL,
                                         bg="#1a1a1a", fg="white", highlightthickness=0,
                                         command=lambda v: self.analyze_spectrum())
        self.dispersion_slider.set(1.0)
        self.dispersion_slider.pack(fill=tk.X, padx=5)

        tk.Label(control_panel, text="Emission Threshold:",
                bg="#1a1a1a", fg="white").pack(anchor=tk.W, padx=5)
        self.threshold_slider = tk.Scale(control_panel, from_=0, to=200,
                                        orient=tk.HORIZONTAL, bg="#1a1a1a",
                                        fg="white", highlightthickness=0,
                                        command=lambda v: self.analyze_spectrum())
        self.threshold_slider.set(50)
        self.threshold_slider.pack(fill=tk.X, padx=5)

        tk.Label(control_panel, text="📡 DETECTOR SETTINGS",
                font=("Courier New", 11, "bold"), bg="#1a1a1a", fg="#00ff00").pack(pady=(20, 5))

        self.continuum_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_panel, text="Show Continuum Background",
                      variable=self.continuum_var, command=self.analyze_spectrum,
                      bg="#1a1a1a", fg="white", selectcolor="#2a2a2a").pack(anchor=tk.W, padx=5)

        self.absorption_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_panel, text="Absorption Mode",
                      variable=self.absorption_var, command=self.analyze_spectrum,
                      bg="#1a1a1a", fg="white", selectcolor="#2a2a2a").pack(anchor=tk.W, padx=5)

        self.labels_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_panel, text="Show Element Labels",
                      variable=self.labels_var, command=self.analyze_spectrum,
                      bg="#1a1a1a", fg="white", selectcolor="#2a2a2a").pack(anchor=tk.W, padx=5)

        tk.Label(control_panel, text="Detector Noise:",
                bg="#1a1a1a", fg="white").pack(anchor=tk.W, padx=5)
        self.noise_slider = tk.Scale(control_panel, from_=0, to=50,
                                    orient=tk.HORIZONTAL, bg="#1a1a1a",
                                    fg="white", highlightthickness=0,
                                    command=lambda v: self.analyze_spectrum())
        self.noise_slider.set(5)
        self.noise_slider.pack(fill=tk.X, padx=5)

        tk.Label(control_panel, text="Wavelength Range (nm):",
                bg="#1a1a1a", fg="white").pack(anchor=tk.W, padx=5)
        range_frame = tk.Frame(control_panel, bg="#1a1a1a")
        range_frame.pack(fill=tk.X, padx=5)

        self.min_wavelength = tk.IntVar(value=380)
        self.max_wavelength = tk.IntVar(value=750)
        tk.Entry(range_frame, textvariable=self.min_wavelength, width=5,
                bg="#2a2a2a", fg="white").pack(side=tk.LEFT, padx=2)
        tk.Label(range_frame, text="-", bg="#1a1a1a", fg="white").pack(side=tk.LEFT)
        tk.Entry(range_frame, textvariable=self.max_wavelength, width=5,
                bg="#2a2a2a", fg="white").pack(side=tk.LEFT, padx=2)

        tk.Button(control_panel, text="Update Range",
                 command=self.analyze_spectrum, bg="#2a2a2a", fg="white").pack(pady=5)

        tk.Label(control_panel, text="🔍 DETECTED ELEMENTS",
                font=("Courier New", 11, "bold"), bg="#1a1a1a", fg="#00ff00").pack(pady=(15, 5))

        self.elements_listbox = tk.Listbox(control_panel, bg="#0a0a0a", fg="#00ff00",
                                          font=("Courier New", 9), height=5)
        self.elements_listbox.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(control_panel, text="💾 ACTIONS",
                font=("Courier New", 11, "bold"), bg="#1a1a1a", fg="#00ff00").pack(pady=(15, 5))

        tk.Button(control_panel, text="🔄 Re-Analyze",
                 command=self.analyze_spectrum, bg="#2a5a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)
        tk.Button(control_panel, text="💾 Export Spectrum",
                 command=self.export_spectrum, bg="#2a5a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)
        tk.Button(control_panel, text="📊 Full Resolution",
                 command=self.show_full_resolution, bg="#2a5a2a", fg="white").pack(fill=tk.X, padx=5, pady=2)

        display_panel = tk.Frame(main_content, bg="#050505")
        display_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.spectrum_canvas = tk.Canvas(display_panel, bg="#000000", highlightthickness=0)
        self.spectrum_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.info_label = tk.Label(display_panel, text="Initializing spectrometer...",
                                   bg="#0a0a0a", fg="#00ff00", anchor=tk.W,
                                   font=("Courier New", 9))
        self.info_label.pack(fill=tk.X, padx=5, pady=5)

    def analyze_spectrum(self):
        if not self.source_image:
            return

        try:
            self.info_label.config(text="Analyzing spectrum...")
            self.window.update()

            slit_width = self.slit_width_slider.get()
            exposure_stacks = self.exposure_slider.get()
            dispersion = self.dispersion_slider.get()
            threshold = self.threshold_slider.get()
            continuum = self.continuum_var.get()
            absorption = self.absorption_var.get()
            show_labels = self.labels_var.get()
            noise = self.noise_slider.get()

            min_wl = self.min_wavelength.get()
            max_wl = self.max_wavelength.get()

            img_array = np.array(self.source_image)
            img_height, img_width = img_array.shape
            slit_step = max(1, img_width // (exposure_stacks + 1))

            self.detected_elements = []
            self.elements_listbox.delete(0, tk.END)

            spectrum_height = 600
            spectrum_width = 800
            spectrum_img = np.zeros((spectrum_height, spectrum_width, 3), dtype=np.float32)

            if continuum:
                for y in range(spectrum_height):
                    wavelength = min_wl + (y / spectrum_height) * (max_wl - min_wl)
                    color = np.array(self.wavelength_to_rgb(wavelength), dtype=np.float32)
                    continuum_intensity = 0.3 + 0.2 * np.sin(np.pi * y / spectrum_height)
                    spectrum_img[y, :, :] = color * continuum_intensity

            slit_positions = []
            for i in range(exposure_stacks):
                x_pos = min(i * slit_step + slit_width // 2, img_width - 1)
                if x_pos < img_width:
                    slit_positions.append(x_pos)

            for slit_idx, slit_x in enumerate(slit_positions):
                strip = img_array[:, slit_x:slit_x + slit_width]
                if strip.size == 0:
                    continue
                intensity_profile = np.mean(strip, axis=1)

                for y_pos in range(spectrum_height):
                    img_y = int((y_pos / spectrum_height) * img_height)
                    if img_y >= len(intensity_profile):
                        continue

                    intensity = intensity_profile[img_y]
                    if intensity >= threshold:
                        wavelength = min_wl + (y_pos / spectrum_height) * (max_wl - min_wl)

                        for element, lines in self.ELEMENTAL_LINES.items():
                            if element == 'Fraunhofer': continue

                            for line_wl in lines:
                                wl_diff = abs(wavelength - line_wl) * dispersion
                                if wl_diff < 2.0:
                                    line_color = np.array(self.wavelength_to_rgb(line_wl), dtype=np.float32)
                                    brightness = (intensity / 255.0) * (1.0 - wl_diff / 2.0)
                                    brightness = min(brightness, 1.0)

                                    if absorption:
                                        spectrum_img[y_pos, :, :] *= (1.0 - brightness * 0.8)
                                    else:
                                        line_width = max(1, int(3 - wl_diff))
                                        for offset in range(-line_width, line_width + 1):
                                            y_off = y_pos + offset
                                            if 0 <= y_off < spectrum_height:
                                                offset_brightness = brightness * (1.0 - abs(offset) / (line_width + 1))
                                                spectrum_img[y_off, :, :] = np.maximum(
                                                    spectrum_img[y_off, :, :],
                                                    line_color * offset_brightness
                                                )

                                        if element not in self.detected_elements and brightness > 0.3:
                                            self.detected_elements.append(element)

            if noise > 0:
                noise_array = np.random.randn(*spectrum_img.shape) * noise
                spectrum_img = np.clip(spectrum_img + noise_array, 0, 255)

            spectrum_img = np.clip(spectrum_img, 0, 255).astype(np.uint8)
            result_image = Image.fromarray(spectrum_img)

            self.spectrum_canvas.update()
            canvas_width = self.spectrum_canvas.winfo_width()
            canvas_height = self.spectrum_canvas.winfo_height()

            if canvas_width < 100: canvas_width = 700
            if canvas_height < 100: canvas_height = 500

            scale = min(canvas_width / spectrum_width, canvas_height / spectrum_height)
            disp_w = int(spectrum_width * scale)
            disp_h = int(spectrum_height * scale)

            display_image = result_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
            self.spectrum_photo = ImageTk.PhotoImage(display_image)

            self.spectrum_canvas.delete("all")
            self.spectrum_canvas.create_image(canvas_width // 2, canvas_height // 2,
                                            image=self.spectrum_photo, anchor=tk.CENTER)

            scale_x = canvas_width // 2 + disp_w // 2 + 30
            for wl in range(min_wl, max_wl + 1, 50):
                y = canvas_height // 2 - disp_h // 2 + int((wl - min_wl) / (max_wl - min_wl) * disp_h)
                if 0 <= y < canvas_height:
                    self.spectrum_canvas.create_line(scale_x, y, scale_x + 15, y, fill="white", width=1)
                    self.spectrum_canvas.create_text(scale_x + 25, y, text=f"{wl}", fill="white", font=("Courier New", 8))

            if show_labels and self.detected_elements:
                label_x = canvas_width // 2 - disp_w // 2 + 20
                for idx, element in enumerate(self.detected_elements[:10]):
                    label_y = canvas_height // 2 - disp_h // 2 + 30 + idx * 20
                    self.spectrum_canvas.create_text(label_x, label_y,
                                                    text=f"✓ {element}",
                                                    fill="#00ff00", font=("Courier New", 10, "bold"),
                                                    anchor=tk.W)

            self.elements_listbox.delete(0, tk.END)
            for element in sorted(set(self.detected_elements)):
                self.elements_listbox.insert(tk.END, f"★ {element}")

            mode_str = "ABSORPTION" if absorption else "EMISSION"
            self.info_label.config(
                text=f"{mode_str} MODE | Slits: {len(slit_positions)} | Elements: {len(set(self.detected_elements))} | Range: {min_wl}-{max_wl}nm"
            )

            self.current_spectrum = result_image

        except Exception as e:
            self.info_label.config(text=f"Analysis Error: {str(e)}")

    def export_spectrum(self):
        if hasattr(self, 'current_spectrum'):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
                title="Export Spectrum Analysis"
            )
            if file_path:
                try:
                    self.current_spectrum.save(file_path)
                    messagebox.showinfo("Export Complete", f"Spectrum saved to:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Export Error", f"Failed: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No spectrum to export")

    def show_full_resolution(self):
        if hasattr(self, 'current_spectrum'):
            full_window = tk.Toplevel(self.window)
            full_window.title("Full Resolution Spectrum")
            full_window.geometry("1024x768")
            full_window.configure(bg="black")

            spectrum_array = np.array(self.current_spectrum)
            height, width = spectrum_array.shape[:2]

            canvas = tk.Canvas(full_window, bg="black", highlightthickness=0)
            v_scroll = tk.Scrollbar(full_window, orient=tk.VERTICAL, command=canvas.yview)
            h_scroll = tk.Scrollbar(full_window, orient=tk.HORIZONTAL, command=canvas.xview)

            canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

            canvas.create_window((0, 0), image=ImageTk.PhotoImage(self.current_spectrum), anchor=tk.NW)
            canvas.configure(scrollregion=(0, 0, width, height))


def main():
    root = tk.Tk()
    app = SounderStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()