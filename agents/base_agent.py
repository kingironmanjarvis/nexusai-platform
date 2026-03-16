"""
Base Agent — All agents inherit from this
"""
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from loguru import logger
from core.reasoning.engine import ReasoningEngine
from core.memory.memory import MemorySystem

@dataclass
class AgentResult:
    agent_id: str
    agent_type: str
    task: str
    result: str
    success: bool
    tokens_used: int = 0
    latency_ms: int = 0
    confidence: float = 0.0
    error: Optional[str] = None
    artifacts: List[Dict] = field(default_factory=list)

class BaseAgent:
    agent_type = "base"
    
    def __init__(self, memory: Optional[MemorySystem] = None, model_tier: str = "balanced"):
        self.id = str(uuid.uuid4())[:8]
        self.engine = ReasoningEngine(model_tier=model_tier)
        self.memory = memory or MemorySystem()
        self.task_count = 0
        self.success_count = 0
        self.total_tokens = 0

    def execute(self, task: str, context: Dict = None) -> AgentResult:
        """Execute a task — override in subclasses"""
        raise NotImplementedError

    def _run(self, prompt: str, system: str, task: str, context: Dict = None) -> AgentResult:
        """Core execution wrapper with timing and logging"""
        start = time.time()
        self.task_count += 1
        
        # Add relevant memories to context
        memories = self.memory.recall(task, n_results=3)
        memory_context = ""
        if memories:
            memory_context = "\n\nRelevant context from memory:\n" + "\n".join(
                [f"- {m['content'][:200]}" for m in memories]
            )
            prompt += memory_context

        try:
            result = self.engine.think(prompt, system=system)
            elapsed = time.time() - start
            self.success_count += 1
            self.total_tokens += result["tokens_used"]
            
            # Store result in memory
            self.memory.remember_short(
                f"Task: {task[:100]} | Result: {result['content'][:200]}",
                metadata={"agent": self.agent_type, "task": task[:100]}
            )
            
            return AgentResult(
                agent_id=self.id,
                agent_type=self.agent_type,
                task=task,
                result=result["content"],
                success=True,
                tokens_used=result["tokens_used"],
                latency_ms=round(elapsed * 1000),
                confidence=result["confidence"],
            )
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[{self.agent_type}] Task failed: {e}")
            return AgentResult(
                agent_id=self.id,
                agent_type=self.agent_type,
                task=task,
                result="",
                success=False,
                latency_ms=round(elapsed * 1000),
                error=str(e),
            )

    def get_stats(self) -> Dict:
        return {
            "agent_id": self.id,
            "agent_type": self.agent_type,
            "tasks_executed": self.task_count,
            "success_rate": self.success_count / max(1, self.task_count),
            "total_tokens": self.total_tokens,
        }
