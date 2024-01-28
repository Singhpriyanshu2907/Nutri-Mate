import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import random
import time

load_dotenv()

DB_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.

Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

# Loading the model
def load_llm():
    llm = OpenAI(temperature=0.5)
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Function to handle user queries
def handle_query(query):
    try:
        qa_result = qa_bot()
        response = qa_result({'query': query})
        answer = response["result"]
        print("Answer from handle_query:", answer)  # Add this line to print the answer
        return answer
    except Exception as e:
        print("Error in handle_query:", e)
        return "ERROR"

# Function to handle user queries and generate responses
def respond(message, chat_history=None):
    if chat_history is None:
        chat_history = []  # Initialize chat_history as an empty list if None
    bot_message = handle_query(message)
    chat_history.append(("User", message))
    chat_history.append(("Nutri-Mate", bot_message))
    time.sleep(2)
    # Return chat history as a list of lists
    return [[sender, msg] for sender, msg in chat_history]

# Create Gradio interface with custom styling
iface = gr.Interface(
    fn=respond,  # Use the respond function to handle user queries
    inputs=gr.Textbox(placeholder="Type your message here..."),
    outputs=gr.Chatbot(),
    title="Nutri-Mate",
    description="Hi! This is Nutri-Mate, your Nutrition & Health Assistance Bot.",
    theme="huggingface",
    allow_flagging=False,
    examples=[
        ["What are the benefits of eating fruits and vegetables?"],
        ["Can you suggest some foods high in protein?"],
        ["How does exercise affect mental health?"],
        ["What are the symptoms of vitamin deficiency?"],
        ["Is intermittent fasting good for weight loss?"],
        ["What are the risks of a high-sugar diet?"],
        ["Can you provide some tips for improving sleep quality?"],
    ],
    analytics_enabled=False
)

# Modified msg.submit call to properly pass chat_history
msg = gr.Textbox(placeholder="Type your message here...")  # Define the input component
chatbot = gr.Chatbot()  # Define the output component
clear = gr.ClearButton([msg, chatbot])  # Define the clear button

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)