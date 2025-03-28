import os
from dotenv import load_dotenv
from groqapi import llama_70b_versatile_api_call,deepseek_r1_api_call

load_dotenv()
api = os.getenv("api_key")

# condition = True
# while condition:
#     llama_70b_versatile_api_call(api, input("Ask your question!!"))
#     print('''
    
#     _________________________________task completed_____________________________________
    
#     ''')


from langchain_groq import ChatGroq
import os
from langchain.memory.summary import ConversationSummaryMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

llm = ChatGroq(
    groq_api_key = os.getenv("api_key"),
    model_name="llama-3.3-70b-versatile",
)
memory =   ConversationSummaryMemory(llm=llm)

while True:
    user_input = input("You: ask your question here-")
    chat_history = memory.load_memory_variables({})['history']
    #conversation_history = memory.load_memory_variable({})[user_input]
    print(chat_history,"history of chat-----------------------------")
    messages = [
        ("human",f"Ask your question: \n{user_input}\n\nChat History:\n{chat_history}"),
    ]

    response=llm.invoke(messages)
    print(response.content)
    
    memory.save_context({"input":user_input},{"output": (response.content)})
    with open (r"C:\GEN AI\chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"User: {user_input}\n")
        f.write(f"AI: {response.content}\n")
        f.write("\n")
        f.write("______________________________\n")
        f.write("\n")
        f.write("\n")
# # Ensure the chat history is saved properly
# try:
#     with open(r"C:\GEN AI\chat_history.txt", "a", encoding="utf-8") as f:
#         f.write(f"User: {user_input}\n")
#         f.write(f"AI: {response.content}\n")
#         f.write("\n")
#         f.write("______________________________\n")
#         f.write("\n")
#         f.write("\n")
#     print("Chat history saved successfully.")
# except Exception as e:
#     print(f"An error occurred while saving the chat history: {e}")
