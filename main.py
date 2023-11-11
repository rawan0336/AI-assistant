

import os
import openai
import sys
import random
import streamlit as st

from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise Exception("API key is missing. Set the OPENAI_API_KEY environment variable.")
openai.api_key = api_key


def read_data():#take whats inside data
    text ="" #varنعبي فيه التكست اللي بنسوي ليه summerize
    data_dir = os.path.join(os.getcwd(),"data")# نعرف وين موجود getcwd:get current working directory
    for filename in os.listdir(data_dir):
        with open(os.path.join(data_dir,filename),"r") as f:
            text+=f.read()
    return text

def create_docs(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=200)
    docs = text_splitter.split_text(text)#بترجع لي list من text
    print (f"number of documents are {len(docs)}")#how many documents does he recognize
    return docs

def create_embedding(docs):
    embeddings = OpenAIEmbeddings()
    doc_search=FAISS.from_texts(docs,embeddings)
    return doc_search

def response_chain(doc_search,prompt,LLM):
    from langchain.chains.question_answering import load_qa_chain
    chain= load_qa_chain (llm=LLM,chain_type="stuff")
    docs= doc_search.similarity_search(prompt)
    response = chain.run(input_documents = docs,question=prompt)
    return response

def main():
    st.title("Chat with your data")
    LLM=ChatOpenAI(
        temperature=0.8,
        model_name="gpt-3.5-turbo",
        openai_api_key=openai.api_key,
    )
    placeholders = [
        "What is the name of the instructor of OS course?",
        "What is the mark distribution of MATH course?",
        "How many courses do I have?",
    ]

    question = st.text_input(
        "Ask something about the Syllabus",
        placeholder=random.choice(placeholders),
    )
    if question:
        with st.spinner("Thinking..."):
            response=response_chain(create_embedding(create_docs(read_data())),prompt=question,LLM=LLM)
            st.write("### Answer")
            st.write(response)


if __name__ == "__main__":
    main()
    

