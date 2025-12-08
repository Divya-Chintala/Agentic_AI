import streamlit as st
import pandas as pd
import numpy as np
# API keys
import os 
from dotenv import load_dotenv

# For LLM
from langchain_groq import ChatGroq


# Embedding model
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Type validation
from pydantic import BaseModel,Field

# Vector DB
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Output Parser for desired output structure
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# LangGrapgh Workflow
from langgraph.graph import StateGraph,MessagesState,START,END
from typing import TypedDict,List



load_dotenv()



llm=ChatGroq(model='llama-3.1-8b-instant')
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "wine-rag-pipelin"  
index = pc.Index(index_name)

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



vector_store = PineconeVectorStore(index=index, embedding=embeddings)

class StructuredOutput(BaseModel):
    country:str = Field(description="country name")
    wine:str = Field(description="Wine name")
    price_in_USD:int = Field(description="Wine price")
    wine_variety:str = Field(description="wine variety")
    Reason:str = Field(description="Reasoning for this suggestion in 4-5 lines")

class State(TypedDict):
    question:str
    context:List[str]
    answer:str

graph=StateGraph(State)
parser=PydanticOutputParser(pydantic_object=StructuredOutput)

def retrieve(state:State):
    
    #print(state)

    retriever=vector_store.as_retriever(search_kwargs={"k": 3} )

    retrival_result=retriever.invoke(state['question'])
    result=[]
    for res in retrival_result:
        result.append(res.page_content)


    return {"context":result}

def generate(state:State):
    
    #print(state)
    

    format_instructions = parser.get_format_instructions()

    template= """ You are an assistant for question-answering tasks. Use the following pieces of retrieved context to 
    answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise in format : {format_instructions}
    Question: {question} 
    Context: {context} 
    Answer: """

    prompt=PromptTemplate(
        template=template,
        input_variables=["question","context"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    rag= prompt | llm 
    
    answer=rag.invoke({"question":state['question'],"context":state['context']})
    
    return {'answer':answer.content}

graph.add_node("Retrive",retrieve)
graph.add_node("Generate",generate)
graph.add_edge(START,"Retrive")
graph.add_edge("Retrive","Generate")

rag_pipeline=graph.compile()

st.title("RAG Based Wine Recommendation system")

query=st.text_input(label="Enter your query for wine recommendation")

button=st.button(label="Get Recommendation")



if button:
    result = rag_pipeline.invoke({"question": query})
    context = result['context']
    answer = result['answer']
    if isinstance(answer, str):
        answer = parser.parse(answer) 
    


    st.subheader("Wine Recommendation : ")
    st.markdown(f"**Wine Name:** {answer.wine}")
    # st.markdown(f"**Country:** {answer.country}")
    st.markdown(f"**Wine Variety:** {answer.wine_variety}")
    st.markdown(f"**Price in (USD):** ${answer.price_in_USD}")
    st.markdown(f"**Reason:** {answer.Reason}")

    st.subheader("Retrieved Context :")
    for i, ctx in enumerate(context, 1):
        st.markdown(f"**Context {i}:** {ctx}")