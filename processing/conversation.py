def build_conversation_prompt(
    chat_history,
    user_query,
    context,
    max_turns=3
):
    """
    Builds a prompt for the LLM including the last N turns of conversation, document context, and the new user query.
    """
    # Only keep the last max_turns exchanges (user/assistant pairs)
    history = [
        m for m in chat_history if m["role"] in ("user", "assistant")
    ][-max_turns*2:]  # Each turn is user+assistant

    history_str = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful assistant with access to the following document context.

Document Context:
{context}

Conversation so far:
{history_str}
User: {user_query}
Assistant:"""
    return prompt
