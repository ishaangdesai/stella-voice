import os
import queue
import threading
import time
import collections
import sys
import base64
import io
import wave

import numpy as np
import pandas as pd
import torch
import webrtcvad
from faster_whisper import WhisperModel
import google.generativeai as genai
from kokoro import KPipeline

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# ==============================================================================
# FLASK & SOCKETIO SETUP
# ==============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')

# ==============================================================================
# SETTINGS & GLOBAL VARIABLES
# ==============================================================================
from config import (
    WHISPER_MODEL,
    SAMPLE_RATE,
    FRAME_DURATION,
    SILENCE_TIMEOUT,
    DEFAULT_SPEEDUP,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    CSV_PATH
)

# --- Global Objects ---
audio_queue = queue.Queue()
vad = webrtcvad.Vad(3)

model_whisper = None
pipeline = None
gemini_chat = None
session_thread = None

# ==============================================================================
# INITIALIZATION
# ==============================================================================
def initialize_models():
    """Initializes and loads all required models on startup."""
    global model_whisper, pipeline, gemini_chat

    print("Initializing models...")
    if model_whisper is None:
        print("Loading Whisper model...")
        model_whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")

    if pipeline is None:
        print("Loading Kokoro TTS pipeline...")
        pipeline = KPipeline(repo_id="hexgrad/Kokoro-82M", lang_code="a")

    if gemini_chat is None:
        print("Initializing Gemini API...")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        gemini_chat = model.start_chat()

    print("Models initialized successfully. ✅")
    return model_whisper, pipeline, gemini_chat

# ==============================================================================
# TEXT-TO-SPEECH (TTS) & PLAYBACK
# ==============================================================================
def generate_tts_response(text, speedup_factor=DEFAULT_SPEEDUP):
    """Generates a single audio clip from text."""
    global pipeline
    if pipeline is None:
        _, pipeline, _ = initialize_models()
    generator = pipeline(text, voice="af_heart", speed=speedup_factor)
    for _, _, audio in generator:
        return audio

def play_tts_web(wav, samplerate=24000):
    """Sends audio data to the browser for playback via WebSocket."""
    try:
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        wav_norm = np.int16(wav * 32767)
        with io.BytesIO() as byte_io:
            with wave.open(byte_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(samplerate)
                wav_file.writeframes(wav_norm.tobytes())
            wav_bytes = byte_io.getvalue()
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        socketio.emit('play_audio', {'audio': audio_b64, 'samplerate': samplerate})
    except Exception as e:
        print(f"Error sending audio to client: {e}")

def generate_and_play_tts(text, speedup_factor=DEFAULT_SPEEDUP):
    """Generates and plays TTS audio, updating client status."""
    socketio.emit('update_status', {'status': 'Speaking...'})
    socketio.emit('log_message', {'speaker': 'AI', 'text': text})
    try:
        audio_tensor = generate_tts_response(text, speedup_factor=speedup_factor)
        play_tts_web(audio_tensor)
        duration = len(audio_tensor) / 24000
        socketio.sleep(duration + 0.2)
    except Exception as e:
        print(f"Error in TTS playback: {e}")

def preload_tts(text, key, audio_cache, speedup_factor):
    """Generates TTS audio from known text and stores it in a cache."""
    try:
        audio = generate_tts_response(text, speedup_factor=speedup_factor)
        audio_cache[key] = audio
    except Exception as e:
        print(f"Error preloading TTS for key '{key}': {e}")

def preload_gemini_tts(prompt, key, audio_cache, speedup_factor):
    """Generates a Gemini response and then preloads its TTS audio."""
    try:
        response_text = gemini_chat.send_message(prompt).text.strip()
        if response_text:
            audio = generate_tts_response(response_text, speedup_factor=speedup_factor)
            audio_cache[key] = {'text': response_text, 'audio': audio}
    except Exception as e:
        print(f"Error preloading Gemini TTS for key '{key}': {e}")

# ==============================================================================
# AUDIO INPUT & TRANSCRIPTION
# ==============================================================================
def float_to_pcm(audio):
    return (audio * 32767).astype(np.int16).tobytes()

def transcribe_until_silence(max_wait_seconds=20):
    """The corrected transcription loop with internal buffering."""
    global model_whisper
    if model_whisper is None: model_whisper, _, _ = initialize_models()
    while not audio_queue.empty(): audio_queue.get_nowait()
    frame_len = int(SAMPLE_RATE * FRAME_DURATION / 1000)
    frame_bytes = frame_len * 2
    silence_frames_needed = int(SILENCE_TIMEOUT / (FRAME_DURATION / 1000))
    audio_buffer, ring_buffer = [], collections.deque(maxlen=silence_frames_needed)
    detected_speech, start_time = False, time.time()
    internal_audio_buffer = b''
    socketio.emit('update_status', {'status': 'Listening...'})
    print("Listening for user response...")
    while True:
        if time.time() - start_time > max_wait_seconds:
            print("No speech detected within the time limit.")
            return ""
        while len(internal_audio_buffer) >= frame_bytes:
            pcm_frame = internal_audio_buffer[:frame_bytes]
            internal_audio_buffer = internal_audio_buffer[frame_bytes:]
            is_speech = vad.is_speech(pcm_frame, SAMPLE_RATE)
            if is_speech:
                if not detected_speech:
                    print("Speech detected...")
                    socketio.emit('update_status', {'status': 'Hearing you...'})
                    detected_speech = True
                ring_buffer.clear()
                audio_buffer.append(pcm_frame)
            elif detected_speech:
                ring_buffer.append(pcm_frame)
                if len(ring_buffer) == silence_frames_needed:
                    print("Silence detected, transcribing...")
                    socketio.emit('update_status', {'status': 'Thinking...'})
                    all_audio = b''.join(audio_buffer)
                    np_audio = np.frombuffer(all_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    segments, _ = model_whisper.transcribe(np_audio, language="en", vad_filter=True, beam_size=1)
                    text = " ".join(seg.text.strip() for seg in segments)
                    socketio.emit('log_message', {'speaker': 'User', 'text': text})
                    return text.lower()
        try:
            samples = audio_queue.get(timeout=0.1)
            internal_audio_buffer += float_to_pcm(samples)
        except queue.Empty:
            socketio.sleep(0.01)
            continue

# ==============================================================================
# MAIN APPLICATION LOGIC
# ==============================================================================
def run_radiology_session():
    """The main logic loop with corrected dialogue flow."""
    try:
        df = pd.read_csv(CSV_PATH)
        initialize_models()
        score, total_cases = 0, len(df)
        attempted_cases = 0

        generate_and_play_tts("Hello. I hope you're ready for boards cases !", speedup_factor=DEFAULT_SPEEDUP)
        
        for index, row in df.iterrows():
            attempted_cases += 1
            incorrect_attempts = 0 # Track incorrect attempts for the current case
            case_id, history, teachings = row["ID"], row["Name (Clinical History)"], row["User_Comment (Teaching Narrative)"]
            
            image_url = f"/static/images/Slide{case_id}.JPG"
            socketio.emit('update_case', {
                'image_url': image_url,
                'score': score,
                'attempted': attempted_cases
            })
            
            generate_and_play_tts(f"{history}.", speedup_factor=DEFAULT_SPEEDUP)

            diagnosed = False
            audio_cache = {}
            correct_text = f"Correct, {teachings}"
            incorrect_text = f"Not quite. The key finding is: {teachings}"
            help_prompt = f"""<<INSTRUCTION>>\nYou are a Radiology Attending testing a Resident. Extract the fracture location in the case below. Without stating the diagnosis, give a 1-sentence clue which bone the fracture is located, such as 'Take a closer look at the scaphoid' or 'What do you think of the scaphoid' or 'The pain is at the scaphoid'\n\n<<CASE>>\n{teachings}."""

            correct_thread = threading.Thread(target=preload_tts, args=(correct_text, 'correct', audio_cache, DEFAULT_SPEEDUP))
            incorrect_thread = threading.Thread(target=preload_tts, args=(incorrect_text, 'incorrect', audio_cache, DEFAULT_SPEEDUP))
            help_thread = threading.Thread(target=preload_gemini_tts, args=(help_prompt, 'help', audio_cache, DEFAULT_SPEEDUP))
            correct_thread.start(); incorrect_thread.start(); help_thread.start()
            
            user_diagnosis = ""
            while not diagnosed:
                user_diagnosis = transcribe_until_silence()
                print(f"You said: '{user_diagnosis}'")

                if not user_diagnosis: continue
                if any(cmd in user_diagnosis for cmd in ["quit", "end session"]): break
                if any(cmd in user_diagnosis for cmd in ["repeat", "history"]):
                    generate_and_play_tts(f"{history}.", speedup_factor=DEFAULT_SPEEDUP)
                    continue
                if "help" in user_diagnosis or "clue" in user_diagnosis:
                    help_thread.join()
                    preloaded_help = audio_cache.get('help')
                    if preloaded_help and preloaded_help.get('audio') is not None:
                        play_tts_web(preloaded_help['audio'])
                        duration = len(preloaded_help['audio']) / 24000
                        socketio.sleep(duration + 0.2)
                    else:
                        print("Help preloading failed. Generating synchronously.")
                        help_response = gemini_chat.send_message(help_prompt).text.strip()
                        generate_and_play_tts(help_response, speedup_factor=DEFAULT_SPEEDUP)
                    continue
                
                if "diagnosis" in user_diagnosis:
                    feedback_prompt = f"""<<INSTRUCTION>>\nYou are a Radiology Attending testing a Resident on a test case. Extract the diagnosis and location of the fracture in the case below. Then extract the diagnosis and location in the Resident response. Allow for transcription errors in words that may sound alike, is the Resident response correct? \n\n<<TEST CASE>>\n'{teachings}'\n\n<<RESIDENT RESPONSE>>\n'{user_diagnosis}'\n\nAnswer only with YES or NO."""
                    feedback = gemini_chat.send_message(feedback_prompt).text.strip().upper()

                    print(f"Gemini feedback: {feedback}")
                    if "YES" in feedback:
                        diagnosed = True # Correct answer, so the case is finished.
                        score += 1
                        socketio.emit('update_case', {'score': score, 'attempted': attempted_cases})
                        
                        correct_thread.join()
                        audio = audio_cache.get('correct')
                        if audio is not None:
                            play_tts_web(audio)
                            duration = len(audio) / 24000
                            socketio.sleep(duration + 0.2)
                        else:
                            generate_and_play_tts(correct_text)
                    else:
                        # --- MODIFIED LOGIC FOR INCORRECT ANSWERS ---
                        incorrect_attempts += 1
                        if incorrect_attempts == 1:
                            # FIRST incorrect attempt. Give a clue and listen again.
                            print("First incorrect attempt. Giving a clue.")
                            socketio.emit('log_message', {'speaker': 'AI', 'text': "(Incorrect. Here's a clue...)"})
                            
                            help_thread.join()
                            preloaded_help = audio_cache.get('help')
                            if preloaded_help and preloaded_help.get('audio') is not None:
                                play_tts_web(preloaded_help['audio'])
                                duration = len(preloaded_help['audio']) / 24000
                                socketio.sleep(duration + 0.2)
                            else:
                                print("Help preloading failed. Generating synchronously.")
                                help_response = gemini_chat.send_message(help_prompt).text.strip()
                                generate_and_play_tts(help_response, speedup_factor=DEFAULT_SPEEDUP)
                            # 'diagnosed' remains False, so the loop will ask for input again.
                        else:
                            # SECOND incorrect attempt. Give the answer and move on.
                            diagnosed = True # Case is finished after the second failed attempt.
                            print("Second incorrect attempt. Revealing answer and moving on.")
                            
                            incorrect_thread.join()
                            audio = audio_cache.get('incorrect')
                            if audio is not None:
                                play_tts_web(audio)
                                duration = len(audio) / 24000
                                socketio.sleep(duration + 0.2)
                            else:
                                generate_and_play_tts(incorrect_text)

            if any(cmd in user_diagnosis for cmd in ["quit", "end session"]): break
        
        generate_and_play_tts(f"Session complete. Your final score is {score} out of {total_cases}. Great work!", speedup_factor=DEFAULT_SPEEDUP)
        socketio.emit('update_status', {'status': 'Session Ended'})
        print("\n--- Session Ended ---")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        socketio.emit('log_message', {'speaker': 'System', 'text': f"A critical error occurred: {e}"})

# ==============================================================================
# SOCKET.IO EVENT HANDLERS & FLASK ROUTES
# ==============================================================================
@socketio.on('connect')
def handle_connect(): print('Client connected')

@socketio.on('disconnect')
def handle_disconnect(): print('Client disconnected')

@socketio.on('start_session')
def handle_start_session():
    global session_thread
    if session_thread is None or not session_thread.is_alive():
        session_thread = socketio.start_background_task(target=run_radiology_session)

@socketio.on('audio_stream')
def handle_audio_stream(data):
    audio_queue.put(np.array(data, dtype=np.float32))

@app.route('/')
def index(): 
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask server...")
    socketio.run(app, debug=False)