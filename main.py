from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import json
import asyncio

app = FastAPI(title="Plant Chatbot API")

class Query(BaseModel):
    query: str

class PlantChatbot:
    def __init__(self, preprocessed_data_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer(
            model_name_or_path="/app/models/sentence_transformer",
            device=self.device
        )
        
        print("Loading data...")
        self.load_data(preprocessed_data_path)
        
        print("Loading Qwen tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/app/models/qwen",
            trust_remote_code=True,
            local_files_only=True
        )
        
        print("Loading Qwen model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "/app/models/qwen",
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Initialization complete!")

    def load_data(self, preprocessed_data_path: str):
        # [Previous load_data implementation remains the same]
        df = pd.read_csv(preprocessed_data_path)
        
        def parse_embedding(embedding_str):
            try:
                embedding_str = embedding_str.strip()
                if embedding_str.startswith('[') and embedding_str.endswith(']'):
                    embedding_str = embedding_str[1:-1]
                return np.fromstring(embedding_str, sep=',')
            except:
                print(f"Error parsing embedding: {embedding_str[:100]}...")
                return None

        df["embedding"] = df["embedding"].apply(parse_embedding)
        df = df.dropna(subset=['embedding'])
        
        self.chunks_data = df.to_dict(orient="records")
        
        embeddings_array = np.stack(df["embedding"].values)
        self.embeddings = torch.tensor(
            embeddings_array,
            dtype=torch.float32
        ).to(self.device)

    def retrieve_relevant_chunks(self, query: str, n_chunks: int = 5) -> List[Dict]:
        # [Previous retrieve_relevant_chunks implementation remains the same]
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False
        ).to(self.device)
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        scores = util.dot_score(query_embedding, self.embeddings)[0]
        _, indices = torch.topk(scores, k=min(n_chunks, len(self.chunks_data)))
        indices = indices.cpu().numpy()
        
        return [
            {
                "sentence_chunk": self.chunks_data[i]["sentence_chunk"],
                "Reference_plant_name": self.chunks_data[i]["Reference_plant_name"],
                "Reference_plant_link": self.chunks_data[i]["Reference_plant_link"]
            }
            for i in indices
        ]

    def format_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        # [Previous format_prompt implementation remains the same]
        context = "- " + "\n- ".join([chunk["sentence_chunk"] for chunk in context_chunks])
        
        prompt = f"""<|im_start|>system
You are a knowledgeable plant expert. Provide helpful and accurate information about plants based on the given context.
<|im_end|>
<|im_start|>user
Based on the following context about plants, please answer the query.
Please be specific and detailed in your response, using only the information provided in the context.
If you cannot answer the question based on the provided context, please say so.

Context:
{context}

User query: {query}
<|im_end|>
<|im_start|>assistant
"""
        return prompt

    async def generate_response(self, query: str) -> Dict:
        try:
            context_chunks = self.retrieve_relevant_chunks(query)
            prompt = self.format_prompt(query, context_chunks)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            max_length = input_ids.shape[1] + 512
            generated_text = ""

            with torch.no_grad():
                generated = input_ids
                
                while generated.shape[1] < max_length:
                    outputs = self.model.forward(
                        input_ids=generated,
                        max_new_tokens=64,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.8,
                        repetition_penalty=1.05,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
                    
                    if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                        break
                        
                    generated = torch.cat([generated, next_token.T], dim=1)
                    new_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                    
                    if new_text:
                        generated_text += new_text
            
            return {
                "Response_text": generated_text,
                "context_items": context_chunks
            }
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {
                "Response_text": "I apologize, but I encountered an error while processing your query. Please try again.",
                "context_items": []
            }

# Initialize chatbot at startup
chatbot = PlantChatbot("updated_plant_data_chunks_and_embeddings.csv")

@app.post("/chat")
async def chat(query: Query):
    """Endpoint for chat interactions"""
    try:
        response = await chatbot.generate_response(query.query)
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Plant Chatbot API is running. Use /chat endpoint for queries."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)