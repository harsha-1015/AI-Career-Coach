from .User_query_service import userQuery

        
    
class Retrival:
    def __init__(self,index,query):
        self.index=index
        self.querVector=self._get_user_query_vector(query)
        self.retrived_info=self._get_info_from_vectorDB(self.querVector)
        
    def _get_info_from_vectorDB(self,query_vector)->list:
        res = self.index.query(vector=query_vector, top_k=10,include_values=False,include_metadata=True  )
        info=[]
        for i in range(len(res['matches'])):
            text=res['matches'][i].metadata.get('text')
            info.append(text)
        return info
        
    def _get_user_query_vector(self,query)->list:
        user_query=userQuery(query)
        query_vector=user_query.queryVector
        return query_vector


