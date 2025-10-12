
from .embedding_service import Embed

embed=Embed()
class userQuery:
    
    def __init__(self,query):
        self.query=query
        self.embeddings=self._get_embeddings()
        self.queryVector=self._get_user_query_embedding()
    def _get_embeddings(self):
        return embed.embeddings
    
    def _get_user_query_embedding(self):
        user_vector=self.embeddings.embed_query(self.query)
        return user_vector

    