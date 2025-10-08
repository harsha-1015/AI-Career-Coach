from ..config import GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import InferenceClient
class LLM:
    def __init__(self):
        self.llm=self._get_llm()
        
    def _get_llm(self):
        llm=ChatGoogleGenerativeAI(
            model='gemini-2.0-flash',
            api_key=GOOGLE_API_KEY
            )
        return llm


    

        