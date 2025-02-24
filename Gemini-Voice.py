import os
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
import threading

# Configure Gemini API
genai.configure(api_key="")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)  # Increase speaking speed

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Specify the model name
model_name = "models/gemini-1.5-pro-002"

def speak(text):
    engine.say(text)
    engine.runAndWait()

while True:
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            user_input = recognizer.recognize_google(audio)
            print(f"User said: {user_input}")

            # Generate content using the selected model
            response = genai.generateContent(
                model=model_name,
                prompt=user_input,
                max_output_tokens=50,  # Limit response length
                temperature=0.7
            )
            bot_response = response.get("content", "")  # Adjust key based on response structure
            print(f"Bot: {bot_response}")

            # Speak response in parallel
            thread = threading.Thread(target=speak, args=(bot_response,))
            thread.start()

        except sr.WaitTimeoutError:
            print("Timeout occurred. Please speak louder or try again.")
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            print(f"Error: {e}")
