import os
from openai import OpenAI

def get_client():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    return  OpenAI(api_key=openai_api_key)
