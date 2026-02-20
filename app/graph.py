from typing_extensions import TypedDict
from typing import Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.constants import START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from app.mem_config import mem_client
from app.prompt import prompt
from app.database.tools import (book_appointment,list_appointments,cancel_appointment,get_current_date_time)

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

available_tools = [
    get_current_date_time,
    book_appointment,
    list_appointments,
    cancel_appointment,
]

llm_with_tools = llm.bind_tools(tools=available_tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):

    last_message = state["messages"][-1]
    user_message = last_message.content

    if len(user_message.split()) < 1:
        memories = []
    else:
        relevant_memories = mem_client.search(
            query=user_message,
            user_id="raj"
        )

        memories = [
            f"ID: {mem.get('id')} Memory: {mem.get('memory')}"
            for mem in relevant_memories.get("results", [])
        ]

    system_prompt = prompt(memories)

    response_chunks = []

    for chunk in llm_with_tools.stream([system_prompt] + state["messages"]):
        if chunk.content:
            #print(chunk.content, end="", flush=True)
            response_chunks.append(chunk.content)

    final_response = "".join(response_chunks)
    return {"messages": [AIMessage(content=final_response)]}

# tool Node
tool_node = ToolNode(tools=available_tools)

# graph Builder
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")

# Conditional edge → if tool is called
graph_builder.add_conditional_edges("chatbot", tools_condition)

# After tool runs → go back to chatbot
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()