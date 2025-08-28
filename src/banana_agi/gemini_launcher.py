import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set GEMINI_API_KEY in your .env file")
        return
    
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents="Explain how AI works in a few words",
    )
    print(response.text)


if __name__ == "__main__":
    main()
