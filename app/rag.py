from .database import VectorDatabase
import ollama
from typing import Tuple, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

class RAG:
    def __init__(self,db_connection_string: str):
        self.ollama_client = ollama.Client()
        self.db = VectorDatabase(db_connection_string)
        
    async def initialize(self):
        """
        Initialize the database connection
        """
        await self.db.initialize()
        
    async def close(self):
        """
        Close the database connection
        """
        await self.db.close()
        
    async def generate_response(self, prompt: str, user_provider_uid: str, conversation_id: str = None):
        try:
            # Create embeddings for the prompt
            prompt_embedding = await self._generate_embeddings(prompt)
            
            # Get conversation history if conversation_id is provided
            conversation_context = ""
            if conversation_id:
                history = await self.db.get_conversation_history(conversation_id)
                conversation_context = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in history
                ])
            
            # Search for similar documents in the vector database
            similar_documents = await self.db.search_similar(
                query_embedding=prompt_embedding,
                user_provider_uid=user_provider_uid,
                query_text=prompt  # Use the prompt as text search query
            )
            
            # Create context from chunk texts and their metadata
            context_parts = []
            for doc in similar_documents:
                chunk_info = f"Chunk {doc['chunk_index']} from document {doc['document_id']}:\n{doc['chunk_text']}"
                context_parts.append(chunk_info)
            
            # Combine all context
            context = "\n\n".join([
                "Previous conversation:",
                conversation_context,
                "\nRelevant documents:",
                "\n\n".join(context_parts)
            ])
            
            # Generate response using the base model
            response = self.ollama_client.generate(
                model="llama3.2",
                prompt=prompt,
                system=context
            )
            
            # Store the conversation if conversation_id is provided
            if conversation_id:
                # Store user message
                user_message_id = await self.db.add_message(conversation_id, "user", prompt, prompt_embedding)
                
                # Store assistant response
                assistant_message_id = await self.db.add_message(conversation_id, "assistant", response.get("response", ""))
                
                return {
                    "response": response.get("response", ""),
                    "user_message_id": user_message_id,
                    "assistant_message_id": assistant_message_id
                }
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    


    async def _generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for the text using Ollama's nomic-embed-text model.
        This model is specifically designed for text embeddings and provides better quality
        embeddings for document similarity search.
        """
        try:
            # Split text into chunks if needed (Ollama has a context window)
            max_chunk_length = 8192  # nomic-embed-text has a larger context window
            text_chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            
            # Generate embeddings for each chunk
            all_embeddings = []
            for chunk in text_chunks:
                # Ollama client is synchronous, no need to await
                response = self.ollama_client.embeddings(
                    model="nomic-embed-text",
                    prompt=chunk
                )
                all_embeddings.append(response['embedding'])
            
            # Average the embeddings if we have multiple chunks
            if len(all_embeddings) > 1:
                return np.mean(all_embeddings, axis=0).tolist()
            return all_embeddings[0]
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Ollama: {str(e)}")
            raise

