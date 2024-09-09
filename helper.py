from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import  PromptTemplate
from langchain.chains import LLMChain
import getpass
import keys
os.environ["GOOGLE_API_KEY"] = keys.gemini

def youtube_url_to_db(url):
    loader = YoutubeLoader.from_youtube_url(url)
    transcript = loader.load()
    
    text_transcript = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 100)
    text_transcript = text_transcript.split_documents(transcript)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(text_transcript , embedding=embeddings )
    return db

def get_resp_query(db , query , k = 10):
    docs = db.similarity_search(query , k = k)
    docs_page_content = ''.join([d.page_content for d in docs])
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    prompt = PromptTemplate(
        input_variables = ["question" , "docs"],
        template="""
                You are a helpful assistant that that can answer questions about youtube videos 
                based on the video's transcript.
                
                Answer the following question: {question}
                By searching the following video transcript: {docs}
                
                Only use the factual information from the transcript to answer the question.
                
                If you feel like you don't have enough information to answer the question, say "I don't know".
                
                Your answers should be verbose and detailed.
                """,   
    )
    chain = LLMChain(llm = llm ,prompt = prompt)
    response = chain.run(question = query , docs =docs_page_content)
    response = response.replace("\n" , "")
    return response