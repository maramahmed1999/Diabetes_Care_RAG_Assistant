"""
Generation — builds a grounded prompt from the retrieved contexts
and calls the local Ollama LLM to produce the final answer.
"""

import ollama

from src import config
from .retriever import hybrid_search


def build_prompt(query: str, contexts: list[dict]) -> str:
    context_str = "\n".join(
        f"--- Source [{i}] ---\n{ctx['text']}"
        for i, ctx in enumerate(contexts, 1)
    )

    return f"""You are a precise medical assistant. Your task is to answer the user's question based ONLY on the provided context below.
If the answer cannot be found in the context, you must state: "I cannot answer this based on the provided documents."
Do NOT use any outside knowledge or make assumptions. Always cite the source number (e.g., [1]) when stating facts.

Context:
{context_str}

User Question: {query}
Answer:"""


def generate_rag_response(query: str, top_k: int = config.TOP_K) -> tuple[str, list[dict]]:
    """
    Runs the full retrieval-augmented generation flow for a single
    user query. Returns (answer_text, sources_used).
    """
    contexts = hybrid_search(query, top_k=top_k)

    if not contexts:
        return "No relevant information found in the database.", []

    prompt = build_prompt(query, contexts)

    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You are a reliable and strict grounded RAG assistant."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0},
    )

    return response["message"]["content"], contexts


if __name__ == "__main__":
    user_query = "What are the common symptoms of type 2 diabetes?"
    answer, sources = generate_rag_response(user_query)

    print("\n" + "=" * 50)
    print("FINAL GROUNDED ANSWER:")
    print("=" * 50)
    print(answer)
