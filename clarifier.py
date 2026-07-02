def clarification_agent(state: dict) -> dict:
    intent = state.get("intent", {})

    if not intent.get("genre"):
        state["needs_clarification"] = True
        state["clarification_question"] = (
            "What genre are you in the mood for?"
        )

    elif not intent.get("scale"):
        state["needs_clarification"] = True
        state["clarification_question"] = (
            "Do you prefer indie or AAA games?"
        )

    return state
