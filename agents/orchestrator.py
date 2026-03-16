"""
Meta-Controller / Orchestrator — Routes tasks, manages agent lifecycle, coordinates execution
"""
import time
import uuid
import json
from typing import Dict, List, Any, Optional
from loguru import logger
from core.memory.memory import MemorySystem
from core.planner.planner import Planner, Task
from agents.coding.coding_agent import CodingAgent
from agents.research.research_agent import ResearchAgent
from agents.reviewer.reviewer_agent import ReviewerAgent
from agents.optimizer.optimizer_agent import OptimizerAgent
from core.reasoning.engine import ReasoningEngine

class Orchestrator:
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.memory = MemorySystem()
        self.planner = Planner()
        self.engine = ReasoningEngine(model_tier="balanced")
        
        # Agent pool
        self.agents = {
            "coding": CodingAgent(memory=self.memory),
            "research": ResearchAgent(memory=self.memory),
            "reviewer": ReviewerAgent(memory=self.memory),
            "optimizer": OptimizerAgent(memory=self.memory),
        }
        
        self.execution_history: List[Dict] = []
        self.session_id = str(uuid.uuid4())[:12]
        logger.info(f"[Orchestrator] Initialized. Session: {self.session_id}")

    def run(self, goal: str, stream_callback=None) -> Dict[str, Any]:
        """Main entry point — take a goal, plan it, execute it, return results"""
        run_id = str(uuid.uuid4())[:8]
        start = time.time()
        logger.info(f"[Orchestrator] Run {run_id}: {goal[:80]}")
        
        # Step 1: Classify intent
        intent = self._classify_intent(goal)
        
        # Step 2: Plan
        if stream_callback:
            stream_callback({"type": "planning", "message": f"Planning: {goal[:60]}..."})
        
        tasks = self.planner.decompose(goal)
        
        if stream_callback:
            stream_callback({"type": "tasks_ready", "tasks": [t.to_dict() for t in tasks]})
        
        # Step 3: Execute tasks
        results = []
        for i, task in enumerate(tasks):
            if stream_callback:
                stream_callback({
                    "type": "task_start",
                    "task_id": task.id,
                    "title": task.title,
                    "agent": task.assigned_agent,
                    "progress": i / len(tasks)
                })
            
            self.planner.update_task_status(task.id, "running")
            agent = self.agents.get(task.assigned_agent, self.agents["research"])
            result = agent.execute(task.description, context={"goal": goal, "task_index": i})
            
            if result.success:
                self.planner.update_task_status(task.id, "done", result=result.result)
            else:
                self.planner.update_task_status(task.id, "failed", error=result.error)
            
            results.append({
                "task_id": task.id,
                "task_title": task.title,
                "agent": task.assigned_agent,
                "success": result.success,
                "result": result.result,
                "tokens_used": result.tokens_used,
                "latency_ms": result.latency_ms,
                "confidence": result.confidence,
            })
            
            if stream_callback:
                stream_callback({
                    "type": "task_done",
                    "task_id": task.id,
                    "success": result.success,
                    "result_preview": result.result[:200],
                    "progress": (i + 1) / len(tasks)
                })
        
        # Step 4: Synthesize final response
        synthesis = self._synthesize(goal, results)
        elapsed = time.time() - start
        
        # Step 5: Store in memory
        self.memory.remember_long(
            f"Goal: {goal}\nOutcome: {synthesis[:300]}",
            metadata={"run_id": run_id, "success_rate": sum(1 for r in results if r["success"]) / max(1, len(results))}
        )
        
        final = {
            "run_id": run_id,
            "session_id": self.session_id,
            "goal": goal,
            "intent": intent,
            "tasks": results,
            "synthesis": synthesis,
            "total_tasks": len(tasks),
            "successful_tasks": sum(1 for r in results if r["success"]),
            "total_tokens": sum(r["tokens_used"] for r in results),
            "elapsed_seconds": round(elapsed, 2),
            "success": sum(1 for r in results if r["success"]) > len(results) * 0.5
        }
        
        self.execution_history.append(final)
        
        if stream_callback:
            stream_callback({"type": "complete", "result": final})
        
        return final

    def chat(self, message: str, history: List[Dict] = None) -> Dict:
        """Simple conversational interface"""
        self.memory.add_conversation_turn("user", message)
        
        system = """You are an advanced AI agent platform assistant. 
You help users accomplish tasks using specialized AI agents.
You are direct, helpful, and technically precise.
When users ask you to DO something, you do it — you don't just explain it."""
        
        result = self.engine.think(
            message,
            system=system,
            history=history or self.memory.get_recent_history(10)
        )
        
        self.memory.add_conversation_turn("assistant", result["content"])
        
        return {
            "response": result["content"],
            "tokens_used": result["tokens_used"],
            "latency_ms": result["latency_ms"],
            "confidence": result["confidence"],
        }

    def _classify_intent(self, goal: str) -> str:
        """Quickly classify what type of task this is"""
        goal_lower = goal.lower()
        if any(w in goal_lower for w in ["code", "build", "implement", "program", "script", "function"]):
            return "coding"
        elif any(w in goal_lower for w in ["research", "analyze", "explain", "what is", "how does"]):
            return "research"
        elif any(w in goal_lower for w in ["automate", "schedule", "run", "execute", "deploy"]):
            return "automation"
        elif any(w in goal_lower for w in ["review", "check", "validate", "test", "verify"]):
            return "review"
        elif any(w in goal_lower for w in ["optimize", "improve", "faster", "better", "performance"]):
            return "optimization"
        return "general"

    def _synthesize(self, goal: str, results: List[Dict]) -> str:
        """Combine all task results into a coherent final answer"""
        if not results:
            return "No results generated."
        
        successful = [r for r in results if r["success"]]
        if not successful:
            return f"All {len(results)} tasks failed. Check individual task errors."
        
        combined = "\n\n".join([
            f"[{r['task_title']}]:\n{r['result'][:800]}"
            for r in successful
        ])
        
        prompt = f"""Synthesize these task results into a single, coherent, complete response.

ORIGINAL GOAL: {goal}

TASK RESULTS:
{combined[:4000]}

Provide a complete, well-structured final answer that:
1. Directly addresses the original goal
2. Incorporates all relevant findings
3. Is ready to present to the user
4. Includes any code, analysis, or artifacts produced"""
        
        result = self.engine.think(prompt, temperature=0.4, max_tokens=3000)
        return result["content"]

    def get_agent_stats(self) -> Dict:
        return {
            "session_id": self.session_id,
            "agents": {name: agent.get_stats() for name, agent in self.agents.items()},
            "memory": self.memory.get_stats(),
            "executions": len(self.execution_history),
            "total_tokens": sum(e["total_tokens"] for e in self.execution_history),
        }
