import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from coach.services.RAG_service import RAG
from pinecone import Pinecone
from coach.config import VECTOR_DB_KEY, VECTOR_DB_HOST
import time
import ast

pc = Pinecone(api_key=VECTOR_DB_KEY)
index=pc.Index(host=VECTOR_DB_HOST)
rag = RAG()

app = Flask(__name__)
allowed_origins = [
    "https://ai-career-coach-frontend-six.vercel.app",
    "https://ai-career-coach-frontend-six.vercel.app/",
    "http://localhost:5173",
    "http://localhost:5173/"
]
CORS(app, origins=allowed_origins)

@app.route('/ask', methods=['POST'])
def ask_coach():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        rag._get_user_query(user_query)
        output=rag._call_llm(index)

        # Try to parse as JSON or Python dict
        try:
            response = json.loads(output)
        except json.JSONDecodeError:
            try:
                response = ast.literal_eval(output)
            except Exception:
                response = {"message": output}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def ask_test():
    try:
        response = "The app is working fine"
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    
    # start=time.time()
    
    # end=time.time()
    # print("\ninitial loading time is=",end-start)
    
    # user=input("enter the query: ")
    # start=time.time()
    
    
    # end=time.time()
    # print("\n")
    # print("time taken=",end-start)
    