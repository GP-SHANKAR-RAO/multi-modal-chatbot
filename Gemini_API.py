# Main Demo STT and TTS
import google.generativeai as genai
import pyttsx3
import speech_recognition as sr

# Configure Gemini API
genai.configure(api_key="")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognizer
recognizer = sr.Recognizer()

while True:
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

        try:
            user_input = recognizer.recognize_google(audio)
            print(f"User said: {user_input}")

            # Send user input to Gemini API
            response = genai.GenerativeModel('gemini-1.5-pro').generate_content(user_input)
            bot_response = response.text
            print(f"Bot: {bot_response}")

            # Speak the response
            engine.say(bot_response)
            engine.runAndWait()

        except Exception as e:
            print(f"Error: {e}")
