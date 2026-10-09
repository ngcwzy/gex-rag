from policy_loader import load_policy_documents
from rag import retrieve_context

documents = load_policy_documents()


def rag_search(question: str) -> str:
    return retrieve_context(question)


try:
    from langchain_core.tools import Tool

    rag_tool = Tool(
        name="university_it_policy_search",
        description=(
            "Search the university IT policy knowledge base. "
            "If the returned context does not contain an explicit answer, "
            "say that you do not know. Do not invent information."
        ),
        func=rag_search,
    )
except Exception:
    rag_tool = rag_search