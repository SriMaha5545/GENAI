import os
from dotenv import load_dotenv
from groqapi import llama_70b_versatile_api_call, deepseek_r1_api_call
from langchain_groq import ChatGroq
from langchain.memory.summary import ConversationSummaryMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from groq import Groq

# Load environment variables
load_dotenv()
api_key = os.getenv("api_key")

if not api_key:
    raise ValueError("API key not found. Please check your .env file.")

# Initialize Groq client with the API key
client = Groq(api_key=api_key)

# Test the Groq client with a completion request
try:
    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    print(completion.choices[0].message)
except Exception as e:
    print(f"Error during Groq client initialization: {e}")

# Initialize LangChain ChatGroq and memory
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.2-11b-vision-preview",
)
memory = ConversationSummaryMemory(llm=llm)

# Main loop for user interaction
while True:
    try:
        # Get user input
        user_input = input("You: ask your question here- ")

        # Get the input image file name
        image_file = input("Enter the image file name or path: ")

        # Load image history from memory
        img_history = memory.load_memory_variables({}).get('history', "No history available.")
        print(img_history, "history of img-----------------------------")

        # Prepare messages for the LLM
        messages = [
            ("human", f"Ask your question: \n{user_input}\n\nImage File: {image_file}\n\nImage History:\n{img_history}"),
        ]

        # Invoke the LLM
        response = llm.invoke(messages)
        print(response.content)

        # Save the context to memory
        memory.save_context({"input": user_input}, {"output": response.content})

        # Save the chat history to a file
        with open(r"C:\GEN AI\img_history.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {user_input}\n")
            f.write(f"Image File: {image_file}\n")
            f.write(f"AI: {response.content}\n")
            f.write("\n")
            f.write("______________________________\n")
            f.write("\n")
        print("Chat history saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")