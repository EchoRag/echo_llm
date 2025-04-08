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
        Initialize the connection pool and create necessary tables
        """
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.connection_string)
            await self._create_tables()
            
    async def _create_tables(self):
        """
        Create necessary tables if they don't exist
        """
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_provider_uid VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_provider_uid) REFERENCES users(provider_uid)
                );

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
                    role VARCHAR(10) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    embedding vector(768)
                );
            ''')
            
    async def create_conversation(self, user_provider_uid: str) -> str:
        """
        Create a new conversation for a user and return its ID
        
        Args:
            user_provider_uid: The provider_uid from the Node.js users table
            
        Returns:
            The new conversation ID
        """
        async with self.pool.acquire() as conn:
            conversation_id = await conn.fetchval('''
                INSERT INTO conversations (user_provider_uid)
                VALUES ($1)
                RETURNING id
            ''', user_provider_uid)
            return str(conversation_id)
            
    async def add_message(self, conversation_id: str, role: str, content: str, embedding: List[float] = None):
        """
        Add a message to a conversation
        """
        async with self.pool.acquire() as conn:
            # Convert embedding list to PostgreSQL vector format if present
            embedding_vector = f"[{','.join(map(str, embedding))}]" if embedding else None
            
            await conn.execute('''
                INSERT INTO conversation_messages (conversation_id, role, content, embedding)
                VALUES ($1, $2, $3, $4)
            ''', conversation_id, role, content, embedding_vector)
            
    async def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent messages from a conversation
        """
        async with self.pool.acquire() as conn:
            messages = await conn.fetch('''
                SELECT role, content
                FROM conversation_messages
                WHERE conversation_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            ''', conversation_id, limit)
            
            return [
                {"role": msg['role'], "content": msg['content']}
                for msg in reversed(messages)  # Reverse to get chronological order
            ]
            
    async def get_user_conversations(self, user_provider_uid: str) -> List[Dict[str, Any]]:
        """
        Get all conversations for a user
        
        Args:
            user_provider_uid: The provider_uid from the Node.js users table
            
        Returns:
            List of conversations with their metadata
        """
        async with self.pool.acquire() as conn:
            conversations = await conn.fetch('''
                SELECT 
                    c.id,
                    c.created_at,
                    c.updated_at,
                    COUNT(cm.id) as message_count
                FROM conversations c
                LEFT JOIN conversation_messages cm ON c.id = cm.conversation_id
                WHERE c.user_provider_uid = $1
                GROUP BY c.id
                ORDER BY c.updated_at DESC
            ''', user_provider_uid)
            
            return [
                {
                    "id": str(conv['id']),
                    "created_at": conv['created_at'],
                    "updated_at": conv['updated_at'],
                    "message_count": conv['message_count']
                }
                for conv in conversations
            ]
            
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