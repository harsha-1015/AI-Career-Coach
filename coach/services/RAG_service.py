from .retrival_service import Retrival
from .llm_service import LLM

llm=LLM()
class RAG:
    def __init__(self):
        self.user_query=None
        
    def _get_user_query(self,query)->None:
        self.user_query=query
    
    def _get_relavent_info(self,index)->list:
           
        retrive=Retrival(index,self.user_query)
        return retrive.retrived_info
    
    def classify_query_hybrid(self,query: str) -> str:
        keywords = ["roadmap", "flowchart", "path", "career path", "journey", "steps", "plan"]
        query_lower = query.lower()
        if any(k in query_lower for k in keywords):
            return "Roadmap"
        elif "how do i become" in query_lower or "progress to" in query_lower:
            return "Roadmap"
        return "notRoadmap"

    
    def _call_llm(self,index):
        relavent_info = self._get_relavent_info(index) 

        
        query_classifier = self.classify_query_hybrid(self.user_query)

        example_json_map = {
            "title": "Data Scientist Roadmap",
            "nodes": [
                {"id": "1", "label": "Learn Python"},
            ],
            "edges": [
                {"from": "1", "to": "3"},
            ]
        }

        if query_classifier == 'Roadmap':
            prompt_Roadmap = f"""return only the Json graph representation in form of {example_json_map} 
                            for the user query {self.user_query} and the relavent information is {relavent_info}"""
            output = llm.llm.invoke(prompt_Roadmap)
            return output.content.strip("```json\n")

        else:
            prompt_notRoadmap = f"""
            Given the question and the knowledge with respect to the question,
            answer in subject to relevate information and it  should be as if your messaging a person, short and correct.
            Question: {self.user_query}
            Relevant Information: {relavent_info}
            """
            output = llm.llm.invoke(prompt_notRoadmap)
            return output.content 

