from pinecone import Pinecone
from .embedding_service import Embed
import os
from ..config import VECTOR_DB_KEY, VECTOR_DB_HOST

class Vectors:
    def __init__(self,index):
        self.index=index
        
        self.vector,self.docs=self._get_vec_docs()
        self.insertVectors=self.create_vectors_upsert()
        self.insert_vectors()
        
        
    def _get_vec_docs(self):
        
        vec,docs=embed._get_parseData()
        return (vec,docs)
    
    def create_vectors_upsert(self):
        vec=self.vector
        doc=self.docs
        vectors=[]
        for i in range(len(vec)):
            v={
                "id":'vec'+str(i),
                'values':vec[i],
                'metadata':{'text':  doc[i].page_content,'producer': doc[i].metadata.get('producer'), 'creator': doc[i].metadata.get('creator'), 'creationdate': doc[i].metadata.get('creationdate'), 'source': doc[i].metadata.get('source'), 'total_pages': doc[i].metadata.get('total_pages'), 'format': doc[i].metadata.get('format'),'page': doc[i].metadata.get('page'), 'source_file': doc[i].metadata.get('source_file'), 'file_type': doc[i].metadata.get('file_type')}
            }
            vectors.append(v)
        return vectors
    
    def insert_vectors(self):
        vectors=self.insertVectors
        self.index.upsert(vectors)
        print(f"vectors inserted to the pinecone DB, number of records inserted are {len(vectors)}")

pc = Pinecone(api_key=VECTOR_DB_KEY)
index=pc.Index(host=VECTOR_DB_HOST)
embed=Embed()  
v=Vectors(index)







