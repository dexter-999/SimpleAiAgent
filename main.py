# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from pydantic import BaseModel
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain.agents import create_agent
# from tools import wiki_tool, search_tools
# from vector import search_company_docs


# load_dotenv()
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash"
#     )


# # from langchain_ollama import ChatOllama

# # llm = ChatOllama(model="qwen3:8b", temperature=0)


# class searchModel(BaseModel):
#     topic:str
#     response:str
#     sources:list[str]
#     tools_used:list[str]
# output_parser = PydanticOutputParser(pydantic_object = searchModel)


# system = ChatPromptTemplate.from_messages(
# 	[
# 		("system",
#     """You are a research assistant.
# And your owner is youssef zounid, But with help llm for google.
# Always use the search tool before answering any question about facts, people, events, dates.
# Don't rely solely on memory to answer.
# If the question is unclear, make a logical assumption (such as the most common explanation), state it clearly, and then search.
# Fill the "Sources" field with the titles of the websites you used, and the "Tools Used" field with the names of the tools you used.
# Answer in the same language and tone as the user's question.
# {format_instructions}"""),
# 		("human","{query}"),
# 	]
# ).partial(format_instructions=output_parser.get_format_instructions()) 
# tools = [wiki_tool, search_tools,search_company_docs]

# agent = create_agent(
# 	model = llm,
# 	tools = tools
# )
# while True:
#     query = input("Enter your question: ")

#     if query == "q":
#         break
#     else:
#         messages = system.invoke({"query":query}).to_messages()
#         response = agent.invoke({"messages":messages})
#         res = response["messages"][-1].content
#         final_output = output_parser.parse(res)
#         print(res)
#         print("--- final response: ---")
#         print(final_output.topic)
#         print(final_output.response)
#         print(final_output.sources)
#         print(final_output.tools_used)

# print("wa baraka, wa salam alaykom")








from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import wiki_tool, search_tools
from vector import search_company_docs


load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.1:8b",)


class searchModel(BaseModel):
    topic: str
    response: str
    sources: list[str]
    tools_used: list[str]


output_parser = PydanticOutputParser(pydantic_object=searchModel)

system = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a research assistant.
And your owner is youssef zounid, But with help llm for google.
Always use the search tool before answering any question about facts, people, events, dates.
Don't rely solely on memory to answer.
If the question is unclear, make a logical assumption (such as the most common explanation), state it clearly, and then search.
Fill the "Sources" field with the titles of the websites you used, and the "Tools Used" field with the names of the tools you used.
Answer in the same language and tone as the user's question.
{format_instructions}"""),
        ("human", "{query}"),
    ]
).partial(format_instructions=output_parser.get_format_instructions())

tools = [search_company_docs,wiki_tool, search_tools ]

agent = create_agent(
    model=llm,
    tools=tools,
)

while True:
    query = input("Enter your question: ")

    # Check the exit condition BEFORE calling the agent,
    # so we don't waste an API call on "q".
    if query.strip().lower() == "q":
        break

    messages = system.invoke({"query": query}).to_messages()
    response = agent.invoke({"messages": messages})

    # AIMessage exposes its text via .content, not .text
    res = response["messages"][-1]

    content = res.content
    if isinstance(content, list):
        res = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        res = content

    try:
        final_output = output_parser.parse(res)
    except Exception as e:
        print(f"Failed to parse structured output: {e}")
        print("Raw response:")
        print(res)
        continue

    print("--- final response: ---")
    print(final_output.topic)
    print(final_output.response)
    print(final_output.sources)
    print(final_output.tools_used)

print("wa baraka, wa salam alaykom")




