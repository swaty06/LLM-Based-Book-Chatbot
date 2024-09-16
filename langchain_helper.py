from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Any, Dict



load_dotenv()  # take environment variables from .env (especially openai api key)

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini model
gemini_model = genai.GenerativeModel(model="gemini-1.5-flash")



# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='faqs.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings, allow_dangerous_deserialization=True)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Use the Gemini model to generate content based on the prompt
    def run_chain(query):
        # Retrieve the relevant context using the retriever
        context_docs = retriever.get_relevant_documents(query)

        # Use the Gemini model to generate the response
        context = "\n".join([doc.page_content for doc in context_docs])
        prompt = PROMPT.format(context=context, question=query)

        # Call the Gemini model to generate the response
        response = gemini_model.generate_content(prompt)
        
        return response.text, context_docs

    return run_chain

if __name__ == "__main__":
    create_vector_db()
     qa_chain = get_qa_chain()
