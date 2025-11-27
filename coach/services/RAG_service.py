from .retrival_service import Retrival
from .llm_service import LLM

llm = LLM()

class RAG():
    def __init__(self):
        self.user_query = None
        
    def _get_user_query(self, query) -> None:
        self.user_query = query
    
    def _get_relavent_info(self, index) -> list:
        retrive = Retrival(index, self.user_query)
        return retrive.retrived_info
    
    def classify_query_hybrid(self, query: str) -> str:
        keywords = ["roadmap", "flowchart", "path", "career path", "journey", "steps", "plan"]
        query_lower = query.lower()
        if any(k in query_lower for k in keywords):
            return "Roadmap"
        elif "how do i become" in query_lower or "progress to" in query_lower:
            return "Roadmap"
        return "notRoadmap"
    
    def _is_followup_question(self, query: str) -> bool:
        """Detect if this is a follow-up question referencing previous context"""
        followup_indicators = [
            "above", "that", "this", "it", "the role", "the position",
            "we discussed", "you mentioned", "you said", "earlier",
            "previous", "same", "also", "what about", "tell me more"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in followup_indicators)

    def _format_history_context(self, history) -> str:
        """Format chat history into a readable context string"""
        if not history:
            return ""
        
        history_text = "\n\nPrevious conversation context:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        
        history_text += "\n"
        return history_text
    
    def _create_context_aware_query(self, query: str, history) -> str:
        """Enhance the query with context from history if it's a follow-up question"""
        if not history or not self._is_followup_question(query):
            return query
        
        # Get the last few exchanges for context
        recent_history = history[-4:] if len(history) > 4 else history
        
        context_query = "Based on our previous conversation:\n"
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_query += f"{role}: {msg['content'][:200]}...\n"  # Limit length
        
        context_query += f"\nCurrent question: {query}"
        return context_query

    def _call_llm(self, index, history=None):
        # Check if this is a follow-up question
        is_followup = self._is_followup_question(self.user_query) if history else False
        
        # For follow-up questions, enhance the query with context
        enhanced_query = self._create_context_aware_query(self.user_query, history) if is_followup else self.user_query
        
        # Get relevant info (use enhanced query for better retrieval on follow-ups)
        if is_followup:
            # Temporarily store original query
            original_query = self.user_query
            self.user_query = enhanced_query
            relavent_info = self._get_relavent_info(index)
            self.user_query = original_query  # Restore original
        else:
            relavent_info = self._get_relavent_info(index)
        
        query_classifier = self.classify_query_hybrid(self.user_query)
        
        # Format history for context
        history_context = self._format_history_context(history) if history else ""

        example_json_map = {
            "title": "Data Scientist Roadmap",
            "nodes": [
                {"id": "1", "label": "Learn Python"},
                {"id": "2", "label": "Master Statistics"},
                {"id": "3", "label": "Study Machine Learning"}
            ],
            "edges": [
                {"from": "1", "to": "2"},
                {"from": "2", "to": "3"}
            ]
        }

        if query_classifier == 'Roadmap':
            prompt_Roadmap = f"""You are a career coach creating a learning roadmap.
            
{history_context}

Current request: {self.user_query}
Relevant information: {relavent_info}

Generate a roadmap in JSON format following this structure exactly:
{example_json_map}

Important:
- Create meaningful nodes with clear learning steps
- Connect nodes with edges showing progression
- Use sequential IDs starting from "1"
- If this is a follow-up question, use the conversation history to understand the context
- Return ONLY the JSON, no additional text or markdown formatting
"""
            output = llm.llm.invoke(prompt_Roadmap)
            return output.content.strip("```json\n").strip("```")

        else:
            prompt_notRoadmap = f"""You are a helpful career coach assistant having a conversation with a user.

{history_context}

Current question: {self.user_query}
Relevant information: {relavent_info}

Instructions:
- Pay close attention to the conversation history above
- If the user asks about something "mentioned above" or "discussed earlier", refer to the previous conversation
- If they use words like "that role", "the position", "it", or "this", look at what was just discussed
- Provide a short, conversational response as if you're messaging a person
- Be concise, relevant, and helpful
- Maintain context from the entire conversation

Answer the current question now:
"""
            output = llm.llm.invoke(prompt_notRoadmap)
            return output.content