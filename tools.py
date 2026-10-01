from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from vector import retrieve

_search = DuckDuckGoSearchResults()
_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)


@tool
def search_tools(query: str) -> str:
    """Search the web (DuckDuckGo) for current events, news, people, and facts."""
    try:
        return _search.run(query)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def wiki_tool(query: str) -> str:
    """Look up a topic on Wikipedia for background information."""
    try:
        return _wiki.run(query)
    except Exception as e:
        return f"Wikipedia failed: {e}"


@tool
def search_company_docs(query: str) -> str:
    """Searches a database of restaurant reviews (customer reviews, ratings, pizza, service, prices, etc.).
    Use this tool first for any questions related to restaurants, customer reviews, food quality, or service,
    before resorting to any external internet searches."""
    results = retrieve.invoke(query)
    if not results:
        return "I did not find any information related to this question in the documents."
    return "\n".join(doc.page_content for doc in results)
