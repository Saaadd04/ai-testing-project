import os
from google.generativeai import GenerativeModel, configure
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    # Load Gemini API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')  # Changed from GROQ_API_KEY
    if api_key is None:
        print('You need to set your GEMINI_API_KEY environment variable')
        exit(1)

    print(f'api key is: {api_key}')
    print("-----------------------------------------")

    # Configure Gemini API
    configure(api_key=api_key)

    # Gemini model (e.g., "gemini-1.5-flash" or "gemini-1.5-pro")
    model = GenerativeModel("gemini-1.5-flash")

    # Messages (Gemini uses a slightly different structure)
    messages = [
        {"role": "user", "parts": [{"text": "My name is Syed Saad"}]},  # System-like instruction as user input
        {"role": "user", "parts": [{"text": "tell about me?"}]},
        # Uncomment and adapt if needed:
        # {"role": "user", "parts": [{"text": "What is Bhopal?"}]},
    ]

    # Streaming response (Gemini supports streaming via generate_content with stream=True)
    stream_response = model.generate_content(
        contents=messages,
        generation_config={
            "temperature": 0.7,  # Matches Grok's default behavior
            "max_output_tokens": 8192  # Matches LLaMA3-70B's context size
        },
        stream=True
    )
    for chunk in stream_response:
        if hasattr(chunk, 'text') and chunk.text is not None:
            print(chunk.text, end='', flush=True)