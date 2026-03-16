"""
Main FastAPI Application — Production API for the AI Agent Platform
"""
import os
import sys
import time
import uuid
import json
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from loguru import logger

# Add platform to path
sys.path.insert(0, '/app/platform')

from agents.orchestrator import Orchestrator
from core.reasoning.engine import ReasoningEngine

# ── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NexusAI Platform",
    description="Production AI Agent Platform — Multi-agent orchestration, real execution, continuous improvement",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──────────────────────────────────────────────────────────────
orchestrator = Orchestrator()
sessions: Dict[str, Dict] = {}
start_time = time.time()

# ── Models ────────────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    goal: str
    session_id: Optional[str] = None
    stream: bool = False

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class CodeRequest(BaseModel):
    code: str
    language: str = "python"

class AgentRequest(BaseModel):
    agent_type: str  # coding, research, reviewer, optimizer
    task: str
    context: Optional[Dict] = None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "platform": "NexusAI",
        "version": "1.0.0",
        "status": "operational",
        "uptime_seconds": round(time.time() - start_time),
        "agents": list(orchestrator.agents.keys()),
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - start_time),
        "groq_connected": bool(os.environ.get("GROQ_API_KEY")),
        "memory_stats": orchestrator.memory.get_stats(),
        "agent_stats": orchestrator.get_agent_stats(),
    }

@app.post("/run")
async def run_goal(req: RunRequest):
    """Execute a goal using the full multi-agent pipeline"""
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")
    
    logger.info(f"[API] /run: {req.goal[:80]}")
    
    try:
        result = orchestrator.run(req.goal)
        return result
    except Exception as e:
        logger.error(f"[API] /run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run/stream")
async def run_goal_stream(req: RunRequest):
    """Execute a goal with streaming progress updates"""
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")
    
    async def event_stream():
        events = []
        done = asyncio.Event()
        
        def callback(event):
            events.append(event)
            if event.get("type") == "complete":
                done.set()
        
        # Run in thread pool to not block
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, orchestrator.run, req.goal, callback)
        
        sent = 0
        while not done.is_set() or sent < len(events):
            while sent < len(events):
                event = events[sent]
                yield f"data: {json.dumps(event)}\n\n"
                sent += 1
            await asyncio.sleep(0.1)
        
        # Make sure task completes
        try:
            await asyncio.wait_for(task, timeout=120)
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout'})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/chat")
async def chat(req: ChatRequest):
    """Conversational interface"""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = {"history": [], "created_at": time.time()}
    
    session = sessions[session_id]
    history = session["history"][-10:]  # Last 10 turns
    
    result = orchestrator.chat(req.message, history=history)
    
    # Update session history
    session["history"].append({"role": "user", "content": req.message})
    session["history"].append({"role": "assistant", "content": result["response"]})
    
    return {
        "session_id": session_id,
        "response": result["response"],
        "tokens_used": result["tokens_used"],
        "latency_ms": result["latency_ms"],
        "confidence": result["confidence"],
    }

@app.post("/agent")
async def run_agent(req: AgentRequest):
    """Run a specific agent directly"""
    agent = orchestrator.agents.get(req.agent_type)
    if not agent:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent. Available: {list(orchestrator.agents.keys())}"
        )
    
    result = agent.execute(req.task, context=req.context)
    return {
        "agent_id": result.agent_id,
        "agent_type": result.agent_type,
        "task": result.task,
        "result": result.result,
        "success": result.success,
        "tokens_used": result.tokens_used,
        "latency_ms": result.latency_ms,
        "confidence": result.confidence,
        "error": result.error,
        "artifacts": result.artifacts,
    }

@app.post("/code/execute")
async def execute_code(req: CodeRequest):
    """Execute code in sandbox"""
    coding_agent = orchestrator.agents["coding"]
    result = coding_agent.write_and_run(req.code, req.language)
    return result

@app.get("/sessions")
async def list_sessions():
    return {
        "sessions": [
            {
                "session_id": sid,
                "created_at": s["created_at"],
                "turns": len(s["history"]) // 2
            }
            for sid, s in sessions.items()
        ]
    }

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

@app.get("/stats")
async def stats():
    return {
        "platform": "NexusAI",
        "uptime_seconds": round(time.time() - start_time),
        "agent_stats": orchestrator.get_agent_stats(),
        "active_sessions": len(sessions),
        "total_executions": len(orchestrator.execution_history),
    }

@app.get("/history")
async def execution_history(limit: int = 10):
    history = orchestrator.execution_history[-limit:]
    return {"executions": history, "total": len(orchestrator.execution_history)}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"deleted": session_id}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting NexusAI Platform on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
