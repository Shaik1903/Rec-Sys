a_k = "AIzaSyAHPpqUignpGcTI1ZfmXfcFcxlpKDtDSrQ"

# Required package:
# pip install google-generativeai

import google.generativeai as genai
import os
from typing import List, Optional

class GeminiAPI:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_text(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text
    
    def start_chat(self):
        return self.model.start_chat(history=[])

class ChatSession:
    def __init__(self, chat):
        self.chat = chat
    
    def send_message(self, message: str) -> str:
        response = self.chat.send_message(message)
        return response.text

def main():
    # Get API key from environment variable
    api_key = a_k
    
    if not api_key:
        print("Please set your Google API key as an environment variable:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return
    
    try:
        # Initialize the API
        gemini = GeminiAPI(api_key)
        
        # Test single generation
        prompt = "Explain quantum computing in simple terms."
        print(f"\nTesting single generation:\nPrompt: {prompt}")
        response = gemini.generate_text(prompt)
        print(f"Response:\n{response}\n")
        
        # Test chat session
        print("\nStarting chat session (type 'quit' to exit):")
        chat_session = ChatSession(gemini.start_chat())
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
            
            response = chat_session.send_message(user_input)
            print(f"Gemini: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()