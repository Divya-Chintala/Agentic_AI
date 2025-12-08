import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, List

# RAG components from your app.py
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate

# RAGAS components
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset # Used by RAGAS to manage data internally

# --- Configuration and Initialization (Copied from your app.py) ---
load_dotenv()

# Initialize LLM and Pinecone for RAG Pipeline
try:
    llm = ChatGroq(model='llama-3.1-8b-instant')
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set.")
    
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "wine-rag-pipelin" 
    
    # --- FIX 2: Correctly extract index names from the Pinecone client response ---
    # pc.list_indexes() returns an object with an 'indexes' attribute (a list of dicts).
    # We map this list to get only the names.
    all_index_objects = pc.list_indexes()
    index_names = [index_info['name'] for index_info in all_index_objects]
    
    if index_name not in index_names:
        raise ValueError(f"Pinecone index '{index_name}' not found. Found: {index_names}")
    # -----------------------------------------------------------------------------
    
    index = pc.Index(index_name)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

except Exception as e:
    print(f"Error initializing RAG components: {e}")
    print("Please ensure your .env file is correct and Pinecone index is active.")
    exit()

# --- RAG Pipeline Components (Copied/Adapted from your app.py) ---

# Pydantic Output Structure for the generator
class StructuredOutput(BaseModel):
    country: str = Field(description="country name")
    wine: str = Field(description="Wine name")
    price_in_USD: int = Field(description="Wine price")
    wine_variety: str = Field(description="wine variety")
    Reason: str = Field(description="Reasoning for this suggestion in 4-5 lines")

# LangGraph State
class State(TypedDict):
    question: str
    context: List[str]
    answer: str

# Define Parser and Prompt
parser = PydanticOutputParser(pydantic_object=StructuredOutput)
format_instructions = parser.get_format_instructions()

def retrieve(state: State):
    """Retrieves relevant contexts from the vector store."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieval_result = retriever.invoke(state['question'])
    
    # Store just the page_content (the text chunks)
    context_chunks = [res.page_content for res in retrieval_result]
    return {"context": context_chunks}

def generate(state: State):
    """Generates the structured answer based on context and question."""
    template = """ You are an assistant for wine recommendation tasks. Use the following pieces of retrieved context to 
    answer the question by recommending a single wine. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise in the following JSON format: {format_instructions}
    Question: {question} 
    Context: {context} 
    Answer: """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "context"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    rag = prompt | llm 
    
    # The output is a str containing the JSON object
    answer = rag.invoke({"question": state['question'], "context": state['context']})
    
    return {'answer': answer.content}

# Build and compile the LangGraph pipeline
graph = StateGraph(State)
graph.add_node("Retrive", retrieve)
graph.add_node("Generate", generate)
graph.add_edge(START, "Retrive")
graph.add_edge("Retrive", "Generate")
rag_pipeline = graph.compile()

# --- RAGAS Evaluation Logic ---

def run_evaluation():
    """Main function to load data, run RAG, and compute RAGAS metrics."""
    print("--- 1. Loading Evaluation Dataset ---")
    try:
        # Load the mock evaluation data (replace with your real dataset)
        with open("mock_eval_data.json", 'r') as f:
            eval_data = json.load(f)
        
        eval_df = pd.DataFrame(eval_data)
        
        # Initialize lists to store the RAG outputs
        generated_answers = []
        retrieved_contexts = []
        
        print(f"--- 2. Running {len(eval_df)} Questions through RAG Pipeline ---")
        
        for index, row in eval_df.iterrows():
            question = row['question']
            
            # Invoke the RAG pipeline
            result = rag_pipeline.invoke({"question": question})
            
            # Store the actual outputs for RAGAS
            retrieved_contexts.append(result['context'])
            generated_answers.append(result['answer']) # This is the raw JSON string
            
            print(f"  Processed Question {index+1}/{len(eval_df)}: {question[:50]}...")
            
        # Add the RAG outputs to the DataFrame
        eval_df['contexts'] = retrieved_contexts
        
        # The 'answer' column must contain the final, unstructured text answer.
        # Since your RAG output is a JSON string, we need to convert it to a readable string.
        # RAGAS does not evaluate the JSON structure, but the content.
        # We will parse the JSON string and use the combined content as the final answer.
        def format_json_answer(json_str):
            try:
                data = json.loads(json_str)
                # Combine the core fields into a single string for RAGAS evaluation
                return (
                    f"Recommended Wine: {data.get('wine', 'N/A')}, "
                    f"Variety: {data.get('wine_variety', 'N/A')}, "
                    f"Price: ${data.get('price_in_USD', 'N/A')}. "
                    f"Reason: {data.get('Reason', 'N/A')}"
                )
            except (json.JSONDecodeError, AttributeError):
                return json_str # Return raw string if parsing fails

        eval_df['answer'] = [format_json_answer(ans) for ans in generated_answers]

        # Convert to HuggingFace Dataset required by RAGAS
        ragas_dataset = Dataset.from_pandas(eval_df)
        
    except Exception as e:
        print(f"An error occurred during data processing or RAG pipeline execution: {e}")
        return

    print("--- 3. Starting RAGAS Metrics Calculation (This may take a minute) ---")

    # Define the metrics
    metrics_to_evaluate = [
        faithfulness,       # Checks if the generated answer is supported by the retrieved context.
        answer_relevancy,   # Checks if the generated answer is relevant to the question.
        context_precision,  # Checks if the retrieved context is relevant to the question.
        context_recall      # Checks if all necessary information for the ground truth is in the context.
    ]

    # Run the evaluation
    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics_to_evaluate,
        llm=llm # Use your Groq LLM to act as the RAGAS judge!
    )

    # Print and display the results
    print("\n" + "="*50)
    print("           RAGAS WINE SYSTEM EVALUATION RESULTS")
    print("="*50)
    print(result)
    
    # Save the detailed results
    results_df = result.to_pandas()
    results_df.to_csv("ragas_wine_results_detailed.csv", index=False)
    print("\nDetailed results saved to ragas_wine_results_detailed.csv")


if __name__ == "__main__":
    run_evaluation()