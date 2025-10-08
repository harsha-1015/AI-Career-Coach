from .retrival_service import Retrival
from .llm_service import LLM
from pinecone import Pinecone
from ..config import VECTOR_DB_KEY, VECTOR_DB_HOST



class RAG:
    def __init__(self,user_query):
        self.user_query=user_query
        self.retrived_info=self._get_relavent_info(user_query)
        
    
    def _get_relavent_info(self,query)->list:
        pc = Pinecone(api_key=VECTOR_DB_KEY)
        index=pc.Index(host=VECTOR_DB_HOST)   
        retrive=Retrival(index,query)
        return retrive.retrived_info
    
    def _call_llm(self)->None:
        prompt = f"""
                Given the question and the knowledge with respect to the question,
                answer as if you're the best career coach in IT sector jobs.
                Remember: the answer should be in 3 lines.

                Question:
                {self.user_query}

                Relevant Information:
                {self.retrived_info}

                Output:
                """
        llm=LLM()
        output=llm.llm.invoke(prompt)
        print(output)
    


    
    

    