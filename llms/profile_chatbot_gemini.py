import os
from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient
from google.generativeai import GenerativeModel, configure


MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")

if __name__ == '__main__':

    api_key = os.getenv('GEMINI_API_KEY')  # Changed from GROQ_API_KEY
    if api_key is None:
        print('You need to set your GEMINI_API_KEY environment variable')
        exit(1)


    configure(api_key=api_key)


    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client['app-dev']
    collection = db['profiles']


    documents = collection.find()


    large_string = ""


    for doc in documents:
        full_name = f"{doc.get('firstName', 'N/A')} {doc.get('lastName', 'N/A')}".strip()
        expertise = doc.get('areaOfExpertise', 'N/A')

        # Build the profile string
        profile_string = f"Name: {full_name}, areaOfExpertise: {doc.get('areaOfExpertise')}, currentLocation: {doc.get('currentLocation')}, Experience: {doc.get('experience')}, highlightedSkills: {doc.get('highlightedSkills')} "
        large_string += f"{profile_string}\n"


    client.close()


    # print(f"Profile Data:\n{large_string}")


    model = GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    You are a chatbot for the Hiretallent project. Use this data to answer the user's query.
    
    Profiles:
    {large_string}

    """

    response = model.generate_content(prompt)
    print(response.text)

    while True:
        user_query = input("Ask a question about the profiles (or type 'exit' to stop):,\nYour Question: ")
        if user_query.lower() == 'exit':
            break
        prompt = f"""
        Profiles:
        {large_string}
    
        User Query:
        {user_query}
        """

        stream_response = model.generate_content(
            contents=prompt,
            generation_config={
                "temperature": 0.7,  # Matches Grok's default behavior
                  # Matches LLaMA3-70B's context size
            },
            stream=True
        )
        print("Gemini's response:")
        for chunk in stream_response:
            if hasattr(chunk, 'text') and chunk.text is not None:
                print(chunk.text, end='', flush=True)