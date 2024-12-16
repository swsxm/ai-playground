from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Define the first prompt template (Instance 1)
template1 = """You are an AI with a curious mind. Talk about something interesting that comes to your mind. Be creative and provide an engaging response."""
prompt1 = ChatPromptTemplate.from_template(template1)

# Define the second prompt template (Instance 2)
template2 = """Instance 1 said: {answer}

Now, continue the conversation by adding more to the topic or introducing a new interesting thought. Feel free to ask questions or explain things further."""
prompt2 = ChatPromptTemplate.from_template(template2)

# Initialize the language models for both instances
model1 = OllamaLLM(model="llama3.1:latest")
model2 = OllamaLLM(model="llama3.1:latest")

# Create the chains for both instances
chain1 = prompt1 | model1
chain2 = prompt2 | model2

# Store the conversation history
history = []

# Number of iterations for the loop (conversation rounds)
iterations = 5

# Loop for the conversation
for i in range(iterations):
    # Step 1: Instance 1 randomly generates a response about anything
    response1 = chain1.invoke({})
    print(f"Instance 1 Response: {response1}")

    # Store Instance 1's response in the history
    history.append(f"Instance 1: {response1}")

    # Step 2: Instance 2 continues the conversation based on Instance 1's output
    response2 = chain2.invoke({"answer": response1})
    print(f"Instance 2 Response: {response2}")

    # Store Instance 2's response in the history
    history.append(f"Instance 2: {response2}")

    # Optionally: If you want to provide a shorter memory, truncate the history
    if len(history) > 10:  # Keep the last 10 exchanges
        history = history[-10:]

# Final conversation history
print("\nFull Conversation History:")
for h in history:
    print(h)
