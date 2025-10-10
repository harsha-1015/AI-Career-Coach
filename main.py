import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from coach.services.RAG_service import RAG
import ast
import sys
from io import StringIO

rag=RAG()
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://ai-career-coach-frontend-six.vercel.app",
            "http://localhost:5173/",
            "http://localhost:517"
        ]
    }
}, supports_credentials=True)

@app.route('/ask', methods=['POST'])
def ask_coach():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Capture output from print()
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        rag._get_user_query(user_query)
        rag._call_llm()

        sys.stdout = old_stdout
        response_str = captured_output.getvalue().strip()

        # Try to parse as JSON or Python dict
        try:
            response = json.loads(response_str)
        except json.JSONDecodeError:
            try:
                response = ast.literal_eval(response_str)
            except Exception:
                # Fallback: treat as plain text response
                response = {"message": response_str}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask-test', methods=['GET'])
def ask_test():
    try:
        response="The app is working fine"
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
