from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper

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
