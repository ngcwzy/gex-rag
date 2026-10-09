import re
from policy_loader import load_policy_documents

documents = load_policy_documents()


def _words(text):
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def retrieve_context(question, top_k=4):
    q_words = _words(question)
    ranked = []

    for doc in documents:
        doc_text = doc.page_content + " " + doc.metadata.get("topic", "")
        d_words = _words(doc_text)
        score = len(q_words & d_words)
        if score > 0:
            ranked.append((score, doc))

    ranked.sort(key=lambda x: x[0], reverse=True)
    picked = [doc for _, doc in ranked[:top_k]]

    if not picked:
        return "No relevant policy text was found."

    parts = []
    for doc in picked:
        parts.append(
            f"[{doc.metadata['policy_id']} | {doc.metadata['topic']}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)