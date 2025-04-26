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
            self.pool = await asyncpg.create_pool(self.connection_string,max_size=5,min_size=2)
            await self._create_tables()
            
    async def _create_tables(self):
        """
        Create necessary tables if they don't exist
        """
        async with self.pool.acquire() as conn:
            await conn.execute('''
                -- Enable required extensions
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pg_trgm;
                
                -- Create tables
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
                    embedding vector(768),
                    upvotes INTEGER DEFAULT 0,
                    downvotes INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS message_votes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    message_id UUID REFERENCES conversation_messages(id) ON DELETE CASCADE,
                    user_provider_uid VARCHAR(255) NOT NULL,
                    vote_type VARCHAR(10) NOT NULL CHECK (vote_type IN ('upvote', 'downvote')),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(message_id, user_provider_uid)
                );

                -- Add user_id to documents_proc if it doesn't exist
                ALTER TABLE documents_proc 
                ADD COLUMN IF NOT EXISTS user_provider_uid VARCHAR(255) NOT NULL DEFAULT 'system';

                -- Add tsvector column for full-text search
                ALTER TABLE document_chunks 
                ADD COLUMN IF NOT EXISTS tsv tsvector 
                GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;

                -- Create indexes
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);

                CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv 
                ON document_chunks USING GIN (tsv);

                CREATE INDEX IF NOT EXISTS idx_documents_proc_user 
                ON documents_proc(user_provider_uid);

                -- Create function to update tsvector
                CREATE OR REPLACE FUNCTION update_document_chunks_tsv()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.tsv := to_tsvector('english', NEW.chunk_text);
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;

                -- Create trigger for tsvector updates
                DROP TRIGGER IF EXISTS update_document_chunks_tsv_trigger ON document_chunks;
                CREATE TRIGGER update_document_chunks_tsv_trigger
                    BEFORE INSERT OR UPDATE ON document_chunks
                    FOR EACH ROW
                    EXECUTE FUNCTION update_document_chunks_tsv();
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
            
    async def add_message(self, conversation_id: str, role: str, content: str, embedding: List[float] = None) -> str:
        """
        Add a message to a conversation
        
        Returns:
            The ID of the created message
        """
        async with self.pool.acquire() as conn:
            # Convert embedding list to PostgreSQL vector format if present
            embedding_vector = f"[{','.join(map(str, embedding))}]" if embedding else None
            
            message_id = await conn.fetchval('''
                INSERT INTO conversation_messages (conversation_id, role, content, embedding)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            ''', conversation_id, role, content, embedding_vector)
            
            return str(message_id)
            
    async def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent messages from a conversation
        """
        async with self.pool.acquire() as conn:
            messages = await conn.fetch('''
                SELECT 
                    id,
                    role, 
                    content,
                    upvotes,
                    downvotes
                FROM conversation_messages
                WHERE conversation_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            ''', conversation_id, limit)
            
            return [
                {
                    "id": str(msg['id']),
                    "role": msg['role'], 
                    "content": msg['content'],
                    "upvotes": msg['upvotes'],
                    "downvotes": msg['downvotes']
                }
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
        user_provider_uid: str,
        query_text: str = None,
        n_results: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using hybrid search (vector + text).
        
        Args:
            query_embedding: Query embedding vector
            user_provider_uid: The user's provider UID to filter documents
            query_text: Optional text query for full-text search
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
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
                # Set number of probes for ivfflat index
                await conn.execute('SET ivfflat.probes = 10;')
                
                if query_text:
                    # Hybrid search with both vector and text
                    results = await conn.fetch('''
                        WITH ranked_results AS (
                            SELECT 
                                dc.id as chunk_id,
                                dc.chunk_text,
                                dc.chunk_index,
                                dc.metadata as chunk_metadata,
                                dp.id as document_id,
                                dp.content as full_content,
                                dp.summary,
                                dp.metadata as document_metadata,
                                1 - (dc.embedding <=> $1::vector(768)) as vector_similarity,
                                ts_rank_cd(dc.tsv, plainto_tsquery($2)) as text_rank
                            FROM document_chunks dc
                            JOIN documents_proc dp ON dc.document_id = dp.id
                            WHERE dc.embedding IS NOT NULL
                                AND dc.tsv @@ plainto_tsquery($2)
                                AND dp.user_provider_uid = $3
                            ORDER BY vector_similarity DESC, text_rank DESC
                            LIMIT $4
                        )
                        SELECT * FROM ranked_results
                        WHERE vector_similarity >= $5
                    ''', query_vector, query_text, user_provider_uid, n_results * 2, similarity_threshold)
                else:
                    # Vector-only search
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
                            1 - (dc.embedding <=> $1::vector(768)) as vector_similarity,
                            0 as text_rank
                        FROM document_chunks dc
                        JOIN documents_proc dp ON dc.document_id = dp.id
                        WHERE dc.embedding IS NOT NULL
                            AND dp.user_provider_uid = $2
                        ORDER BY dc.embedding <=> $1::vector(768)
                        LIMIT $3
                    ''', query_vector, user_provider_uid, n_results)
                
                if not results:
                    logger.warning("No similar documents found")
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
                        "vector_similarity": float(row['vector_similarity']),
                        "text_rank": float(row['text_rank'])
                    }
                    for row in results
                ]
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            raise 
