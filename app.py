import streamlit as st
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load question from file
def load_question(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return "No question found. Make sure 'question.txt' exists."

# Initialize the LLM (use a lighter model if llama3 is too heavy for your RAM)
llm = Ollama(model="llama3")  # You can change to a model that works for your system

# Set up the prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant called Algorythm.PLease optimize whatever code i give in the txt file. Answer the question clearly.\n\nQuestion: {question}\nAnswer:"
)

# Create the LangChain pipeline
chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.title("🤖 LLaMA Q&A Bot")
st.markdown("Automatically answers the question from a `.txt` file.")

# Load and display question
question = load_question("hello.txt")
st.markdown(f"**Question from file:** {question}")

# Run the model and display the answer
with st.spinner("Thinking..."):
    response = chain.run(question)
st.success("Answer:")
st.write(response)
