from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage
load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano")

resp1 = llm.invoke("We are building an AI system for processing medical insurance claims.")
resp2 = llm.invoke("What are the main risks in this system?")

print(resp1.content)
print(resp2.content)

# NOTE:
# The second question ("What are the main risks in this system?")
# depends on context established in the first prompt.
# However, `llm.invoke()` is stateless by default, meaning it does NOT
# remember previous interactions unless conversation history is explicitly passed.
#
# As a result, the model may:
# - Not understand what "this system" refers to
# - Give a generic answer unrelated to medical insurance claims
# - Produce inconsistent outputs across runs
#
# To fix this, you must include prior messages (chat history) 


messages = [
    SystemMessage(content="You are a senior AI architect reviewing production systems."),
    HumanMessage(content="We are building an AI system for processing medical insurance claims."),
    HumanMessage(content="What are the main risks in this system?")
]


resp3 =llm.invoke(messages)
print(resp3.content)

"""
Reflection:

1. Why did string-based invocation fail?
2. Why does message-based invocation work?
3. What would break in a production AI system if we ignore message history?
"""