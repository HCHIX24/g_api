import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded correctly
if not GOOGLE_API_KEY:
    raise ValueError("Google API key is not set in the environment variables.")

# Initialize the Gemini model
llm4 = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Define your query
text = "who am I?"

# Get the response from the model
response = llm4.predict(text)
print(response)
