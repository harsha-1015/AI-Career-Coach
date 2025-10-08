import os
import glob
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
class pdfParser:
    def __init__(self,files_directory):
        """Intilization of the PDF documents

        Args:
            files_directory (_type_): directory of PDF files
        """
        self.files_directory=files_directory
        self.docs=self._get_documents()
        self.chunks=self._make_chunks(self.docs)
        
    def _get_documents(self)->list:
        """Read the PDF files from the directory using PyMuPDFLoader

        Returns:
            list: returns the list of documents from the pdf files, the list contains the loaded pdf files with page_content and metadata
        """
        all_docs=[]
        path_dir=Path(self.files_directory)
        
        files=list(path_dir.glob("*.pdf"))
        
        print(f"Files found {len(files)} to procress")
        for file in files:
            print(f"procressing: {file.name}")
            try:
                loader=PyMuPDFLoader(str(file))
                documents=loader.load()
                
                for doc in documents:
                    doc.metadata['source_file']=file.name
                    doc.metadata['file_type']='pdf'
                all_docs.extend(documents)
                
            except Exception as e:
                print(f"error: {e}")
        print(f"total documents are {len(all_docs)}")
        return all_docs


    def _make_chunks(self,documents)->list:
        """converts teh documents into chunks.

        Args:
            documents (_type_): list of documents that need to be chunked.

        Returns:
            list: list of chunks of the documents.
        """
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n","\n"," ",""]
        )
        split_docs=text_splitter.split_documents(documents)
        print(f"the {len(documents)} documents are split into {len(split_docs)} chunks")
        return split_docs


