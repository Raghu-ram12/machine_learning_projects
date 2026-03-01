import os
from AppOpener import open, close
import webbrowser
import speech_recognition as sr
import pyttsx3
import datetime
import wikipedia
def speak(text):
    
    engine=pyttsx3.init()
    
    engine.setProperty('rate',150)
    print(engine.getProperty('voices'))
    engine.say(text)
    engine.runAndWait()


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source) 
        print("Say something!")
        audio = r.listen(source)
    try:
        # Try Google Speech Recognition (online)
        print("Google Speech Recognition thinks you said: " + r.recognize_google(audio)) # type: ignore
    except Exception as e:
        print(f"Google Speech Recognition error: {e}")
        # Try Sphinx (offline) as fallback
   


apps = {
    "notepad": "notepad",
    "calculator": "calculator",
    "paint": "mspaint",
    "chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge",
    "cmd": "cmd",
    "powershell": "powershell",
    "file explorer": "explorer",
    "control panel": "control",
    "task manager": "taskmgr",
    "vs code": "code",
    "discord": "discord",
    "spotify": "spotify",
    "vlc": "vlc"
}
websites={"google":"https://google.com",
    "youtube":  "https://youtube.com",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "leetcode": "https://leetcode.com",  
    "codechef":"https://codechef.com"}

def open_app(app_name):

    if(app_name in apps):
        try:
           open(apps[app_name],output=False)
           speak(f"opening {app_name}")
        except:
            speak(f"sorry unable to open {app_name}")
    elif app_name in websites:

            try:
                webbrowser.open(websites[app_name])
                speak(f"opening {app_name}")
            except:
                speak(f"sorry unable to open {app_name}")


def close_app(app_name):
    
    if(app_name in apps):
        try:
           close(apps[app_name],output=False,match_closest=True)
           speak(f"closing {app_name}")
        except:
            speak(f"sorry unable to close {app_name}")



def greet():
    
    current_hour = datetime.datetime.now().hour
    
    if current_hour < 12:
        greeting = "Good morning"
    elif 12 <= current_hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
        
    speak(greeting)


def search(text):
    try:
        summary = wikipedia.summary(text)
        speak(summary)
    except Exception as e:
        speak(f"Sorry, I couldn't find information on {text}. Error: {e}")


speak("hello")