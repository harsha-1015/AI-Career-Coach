from coach.services.RAG_service import RAG
  
if __name__ == "__main__":
    user_query=input("Enter your career query: ")
    if user_query.strip():
        rag=RAG(user_query)    
        rag._call_llm()
    else:
        print("please enter a valid query")
    
