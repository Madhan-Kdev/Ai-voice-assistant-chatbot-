import os
import time
import speech_recognition as sr
import pyttsx3
from huggingface_hub import InferenceClient

# ===================== CONFIG =====================
HF_API_KEY = os.getenv("HF_API_KEY")  # set as environment variable
MODEL_NAME = "google/gemma-2-2b-it"
TYPING_DELAY = 0.01  # faster response
USE_VOICE = True     # switch voice on/off
# ==================================================

client = InferenceClient(api_key=HF_API_KEY)

# Initialize TTS engine once (performance optimization)
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 175)

def get_model_response(messages):
    """Stream response from Hugging Face model"""
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=400,
        stream=True
    )

    response_text = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_text += content
            print(content, end="", flush=True)
            time.sleep(TYPING_DELAY)

    print()
    return response_text


def recognize_speech():
    """Convert speech to text"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=6)
            text = recognizer.recognize_google(audio)
            print(f" You: {text}")
            return text
        except:
            print("Could not understand speech")
            return None


def speak(text):
    """Convert text to speech"""
    tts_engine.say(text)
    tts_engine.runAndWait()


def get_user_input():
    """Voice first, text fallback"""
    if USE_VOICE:
        voice_input = recognize_speech()
        if voice_input:
            return voice_input

    return input("\nType your question: ")


# ===================== MAIN LOOP =====================
if __name__ == "__main__":
    print("AI Voice Assistant Started (Say 'exit' to quit)")
    messages = []

    while True:
        user_input = get_user_input()
        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Exiting assistant")
            break

        messages.append({"role": "user", "content": user_input})
        print("\nAssistant: ", end="")
        response = get_model_response(messages)
        messages.append({"role": "assistant", "content": response})

        if USE_VOICE:
            speak(response)
