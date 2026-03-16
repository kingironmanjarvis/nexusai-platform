"""
Planning Engine — Hierarchical task decomposition, goal trees, dependency graphs
"""
import json
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from loguru import logger
from core.reasoning.engine import ReasoningEngine

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    status: str = "pending"  # pending, running, done, failed
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
    assigned_agent: str = ""
    result: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: int = 0

    def to_dict(self):
        d = asdict(self)
        d['subtasks'] = [t.to_dict() if isinstance(t, Task) else t for t in self.subtasks]
        return d

class Planner:
    def __init__(self):
        self.engine = ReasoningEngine(model_tier="balanced")
        self.task_registry: Dict[str, Task] = {}

    def decompose(self, goal: str, context: str = "") -> List[Task]:
        """Break a high-level goal into executable subtasks"""
        system = """You are a master task planner for an AI agent platform.
Break down any goal into concrete, executable subtasks.
Each subtask must be specific enough for an AI agent to execute directly.
Return ONLY valid JSON."""

        prompt = f"""Goal: {goal}
Context: {context}

Decompose this into subtasks. Return JSON array:
[
  {{
    "title": "short task name",
    "description": "exactly what to do",
    "priority": 1-5,
    "assigned_agent": "coding|research|automation|reviewer",
    "dependencies": []
  }}
]

Rules:
- Max 8 subtasks for simple goals, up to 15 for complex ones
- Each subtask must be independently executable
- Order by execution sequence
- Assign the right agent type"""

        result = self.engine.think(prompt, system=system, temperature=0.3)
        
        try:
            # Extract JSON from response
            content = result["content"]
            start = content.find('[')
            end = content.rfind(']') + 1
            json_str = content[start:end]
            raw_tasks = json.loads(json_str)
            
            tasks = []
            for t in raw_tasks:
                task = Task(
                    title=t.get("title", ""),
                    description=t.get("description", ""),
                    priority=t.get("priority", 1),
                    assigned_agent=t.get("assigned_agent", "coding"),
                    dependencies=t.get("dependencies", []),
                )
                self.task_registry[task.id] = task
                tasks.append(task)
            
            logger.info(f"[Planner] Decomposed '{goal[:50]}' into {len(tasks)} tasks")
            return tasks
        except Exception as e:
            logger.error(f"[Planner] Decomposition failed: {e}")
            # Fallback: single task
            fallback = Task(title="Execute goal", description=goal, assigned_agent="coding")
            self.task_registry[fallback.id] = fallback
            return [fallback]

    def estimate_resources(self, tasks: List[Task]) -> Dict[str, Any]:
        """Estimate compute/time resources needed"""
        total_tasks = len(tasks)
        avg_tokens_per_task = 2000
        return {
            "estimated_tasks": total_tasks,
            "estimated_tokens": total_tasks * avg_tokens_per_task,
            "estimated_time_seconds": total_tasks * 5,
            "complexity": "high" if total_tasks > 10 else "medium" if total_tasks > 5 else "low"
        }

    def update_task_status(self, task_id: str, status: str, result: str = None, error: str = None):
        if task_id in self.task_registry:
            self.task_registry[task_id].status = status
            if result:
                self.task_registry[task_id].result = result
            if error:
                self.task_registry[task_id].error = error
