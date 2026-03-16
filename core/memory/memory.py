"""
Memory System — Short-term, Long-term vector memory, RAG retrieval
"""
import os
import json
import time
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except:
    CHROMA_AVAILABLE = False

@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    memory_type: str = "short"  # short, long, episodic

class MemorySystem:
    def __init__(self, persist_dir: str = "/tmp/platform_memory"):
        self.short_term: List[MemoryEntry] = []
        self.max_short_term = 50
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        # Long-term vector memory
        if CHROMA_AVAILABLE:
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name="platform_memory",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("[Memory] ChromaDB initialized for long-term memory")
        else:
            self.collection = None
            logger.warning("[Memory] ChromaDB unavailable, using in-memory only")
        
        # Session memory (dict-based for fast access)
        self.session_memory: Dict[str, Any] = {}
        self.conversation_history: List[Dict] = []

    def remember_short(self, content: str, metadata: Dict = None) -> MemoryEntry:
        """Add to short-term working memory"""
        entry = MemoryEntry(
            content=content,
            metadata=metadata or {},
            memory_type="short"
        )
        self.short_term.append(entry)
        
        # Keep short-term bounded
        if len(self.short_term) > self.max_short_term:
            evicted = self.short_term.pop(0)
            # Promote to long-term
            self._store_long_term(evicted)
        
        return entry

    def remember_long(self, content: str, metadata: Dict = None) -> str:
        """Store in long-term vector memory"""
        entry = MemoryEntry(
            content=content,
            metadata=metadata or {},
            memory_type="long"
        )
        return self._store_long_term(entry)

    def _store_long_term(self, entry: MemoryEntry) -> str:
        """Persist to ChromaDB"""
        if self.collection:
            try:
                self.collection.add(
                    documents=[entry.content],
                    ids=[entry.id],
                    metadatas=[{**entry.metadata, "timestamp": entry.timestamp, "type": entry.memory_type}]
                )
                logger.debug(f"[Memory] Stored long-term: {entry.id}")
                return entry.id
            except Exception as e:
                logger.error(f"[Memory] Long-term store failed: {e}")
        return entry.id

    def recall(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve relevant memories via semantic search"""
        results = []
        
        # Search short-term (keyword match)
        for entry in reversed(self.short_term):
            if query.lower() in entry.content.lower():
                results.append({
                    "content": entry.content,
                    "type": "short_term",
                    "relevance": 0.9,
                    "timestamp": entry.timestamp
                })
        
        # Search long-term vector memory
        if self.collection and self.collection.count() > 0:
            try:
                res = self.collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self.collection.count())
                )
                for doc, meta, dist in zip(
                    res['documents'][0],
                    res['metadatas'][0],
                    res['distances'][0]
                ):
                    results.append({
                        "content": doc,
                        "type": "long_term",
                        "relevance": 1 - dist,
                        "metadata": meta
                    })
            except Exception as e:
                logger.error(f"[Memory] Recall failed: {e}")
        
        # Sort by relevance
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return results[:n_results]

    def add_conversation_turn(self, role: str, content: str):
        """Track conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]

    def get_recent_history(self, n: int = 10) -> List[Dict]:
        return self.conversation_history[-n:]

    def set_session(self, key: str, value: Any):
        self.session_memory[key] = value

    def get_session(self, key: str, default=None) -> Any:
        return self.session_memory.get(key, default)

    def get_stats(self) -> Dict:
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": self.collection.count() if self.collection else 0,
            "conversation_turns": len(self.conversation_history),
            "session_keys": len(self.session_memory)
        }
