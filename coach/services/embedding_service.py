from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from coach.utils.pdf_parser import pdfParser


class Embed:
    def __init__(self):
        
        self.model="sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings=self._get_embeddings()
        
    def _get_embeddings(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.model)
        return embeddings

    def _get_parseData(self):
        parser=pdfParser('data')
        chunks=parser.chunks
        text=[chunk.page_content for chunk in chunks]
        vectors=self.embeddings.embed_documents(text)
        return (vectors, chunks)