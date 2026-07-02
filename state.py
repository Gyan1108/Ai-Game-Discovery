from typing import Optional, Dict, Any

def create_initial_state(query: str) -> Dict[str, Any]:
    return {
        "original_query": query,
        "rewritten_query": "",
        "intent": {},
        "needs_clarification": False,
        "clarification_question": None,
        "results": [],
    }
