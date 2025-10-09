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
    
    def classify_query_hybrid(self,query: str) -> str:
        keywords = ["roadmap", "flowchart", "path", "career path", "journey", "steps", "plan"]
        query_lower = query.lower()
        if any(k in query_lower for k in keywords):
            return "Roadmap"
        elif "how do i become" in query_lower or "progress to" in query_lower:
            return "Roadmap"
        return "notRoadmap"

    
    def _call_llm(self)->None:
        example_json_map={
                            "title": "Data Scientist Roadmap",
                            "nodes": [
                                {"id": "1", "label": "Learn Python"},
                                {"id": "2", "label": "Learn Statistics"},
                                {"id": "3", "label": "Master Pandas & NumPy"},
                                {"id": "4", "label": "Learn Machine Learning"},
                                {"id": "5", "label": "Build Projects"}
                            ],
                            "edges": [
                                {"from": "1", "to": "3"},
                                {"from": "2", "to": "4"},
                                {"from": "3", "to": "5"},
                                {"from": "4", "to": "5"}
                            ]
                        }

        prompt_notRoadmap = f"""
                Given the question and the knowledge with respect to the question,
                answer as if you're the best career coach in IT sector jobs.
                Remember: The answer you provided should be in a conversational/message, 
                short and covers the context and the information you provide should be
                subjected to the Relevent Information otherwise return that you don't know the answer and 
                Question:
                {self.user_query}

                Relevant Information:
                {self.retrived_info}

                Output:
                """
                
        prompt_Roadmap=f"""return only the Json graph representation in form of {example_json_map} 
                        for the user query {self.user_query} and the relavent information is {self.retrived_info}"""
                        
        llm=LLM()
        query_classifier=self.classify_query_hybrid(self.user_query)
        if query_classifier=='Roadmap':
            output=llm.llm.invoke(prompt_Roadmap)
            out=output.content.strip("```json\n")
            print(out)
        else:
            output=llm.llm.invoke(prompt_notRoadmap)
            print(output.content)
            
            
    


    
    

    