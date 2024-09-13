from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Configure Google Gemini API
#genai.configure(api_key=os.getenv('API_KEY'))
#model = genai.GenerativeModel(model_name='gemini-pro')

class GoogleGeminiLLM(LLM):
    def __init__(self, api_key: str, model_name: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
    
    def _call(self, prompt: str, **kwargs: Any) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={
                'temperature': kwargs.get('temperature', 0.1),
                'max_output_tokens': kwargs.get('max_tokens', 800)
            }
        )
        return response['content']  # Ensure this matches the actual API response structure

    def _llm_type(self) -> str:
        return 'GoogleGemini'
llm = GoogleGeminiLLM(api_key=os.getenv('GOOGLE_API_KEY'), model_name='gemini-pro')


# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
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

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
