"""
User configuration variables for the radiology case review application.
"""

# ==============================================================================
# SETTINGS & GLOBAL VARIABLES
# ==============================================================================
WHISPER_MODEL = "large-v3"

#tiny    39 M    ~1 GB    ~32x
#base    74 M    ~1 GB    ~16x

#small    244 M    ~2 GB    ~6x
#medium    769 M    ~5 GB    ~2x
#large-v1    1550 M    ~8 GB    1x (baseline)

#large-v2    1550 M    ~8 GB    1x (baseline)
#large-v3    1550 M    ~8 GB    1x (baseline)

SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
SILENCE_TIMEOUT = 0.4  # s
DEFAULT_SPEEDUP = 1.2

# --- API & Data Paths ---
GEMINI_API_KEY = "AIzaSyCdu0Xbks5qfEgUE4bCmAM9eHA3CpUgRWg"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
CSV_PATH = 'GLASSES_Stella_Emergency_Pilot_SET.csv'
