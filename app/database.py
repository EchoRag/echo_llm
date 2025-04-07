import asyncpg
import logging
from typing import List, Dict, Any
import uuid
import numpy as np
import json

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
        
    async def initialize(self):
        """
        Initialize the connection pool
        """
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.connection_string)
            
    async def close(self):
        """
        Close the connection pool
        """
        if self.pool:
            await self.pool.close()
            self.pool = None
            
    async def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            List of similar documents with their metadata
        """
        try:
            # Validate embedding dimension
            if len(query_embedding) != 768:
                raise ValueError(f"Query embedding must have dimension 768, got {len(query_embedding)}")
                
            # Convert query embedding to PostgreSQL vector format
            query_vector = f"[{','.join(map(str, query_embedding))}]"
            
            async with self.pool.acquire() as conn:
                results = await conn.fetch('''
                    SELECT 
                            dc.id as chunk_id,
                            dc.chunk_text,
                            dc.chunk_index,
                            dc.metadata as chunk_metadata,
                            dp.id as document_id,
                            dp.content as full_content,
                            dp.summary,
                            dp.metadata as document_metadata,
                            1 - (dc.embedding <=> $1::vector(768)) as similarity
                        FROM document_chunks dc
                        JOIN documents_proc dp ON dc.document_id = dp.id
                        WHERE dc.embedding IS NOT NULL
                        ORDER BY dc.embedding <=> $1::vector(768)
                        LIMIT $2
                ''', query_vector, n_results)
                
                if not results:
                    logger.warning("No similar documents found with similarity > 0.5")
                    return []
                
                return [
                    {
                        "chunk_id": str(row['chunk_id']),
                        "chunk_text": row['chunk_text'],
                        "chunk_index": row['chunk_index'],
                        "chunk_metadata": row['chunk_metadata'],
                        "document_id": str(row['document_id']),
                        "full_content": row['full_content'],
                        "summary": row['summary'],
                        "document_metadata": row['document_metadata'],
                        "similarity": float(row['similarity'])
                    }
                    for row in results
                ]
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            raise 