"""
Reasoning Engine — Core LLM interface with Groq
Handles: LLM calls, self-critique, confidence scoring, context management
"""
import os
import json
import time
from typing import Optional, List, Dict, Any
from groq import Groq
from loguru import logger

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODELS = {
    "fast": "llama-3.1-8b-instant",
    "balanced": "llama-3.3-70b-versatile",
    "powerful": "llama-3.3-70b-versatile",   # updated: deepseek decommissioned
}

class ReasoningEngine:
    def __init__(self, model_tier: str = "balanced"):
        self.model = MODELS.get(model_tier, MODELS["balanced"])
        self.history: List[Dict] = []
        self.token_count = 0

    def think(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.time() - start
            content = response.choices[0].message.content
            usage = response.usage

            result = {
                "content": content,
                "model": self.model,
                "tokens_used": usage.total_tokens,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "latency_ms": round(elapsed * 1000),
                "confidence": self._score_confidence(content),
            }
            logger.info(f"[ReasoningEngine] {usage.total_tokens} tokens | {result['latency_ms']}ms | confidence={result['confidence']:.2f}")
            return result
        except Exception as e:
            logger.error(f"[ReasoningEngine] Error: {e}")
            raise

    def critique(self, original_response: str, task: str) -> Dict[str, Any]:
        prompt = f"""You are a self-critique module. Evaluate this response to the task.
TASK: {task}
RESPONSE TO EVALUATE:
{original_response}
Provide JSON with: score (0-10), strengths, weaknesses, improvement, revised_confidence (0.0-1.0)"""
        return self.think(prompt, temperature=0.3, max_tokens=1024)

    def _score_confidence(self, content: str) -> float:
        score = 0.7
        uncertainty_phrases = ["i'm not sure", "i think", "maybe", "possibly", "uncertain"]
        confident_phrases = ["the answer is", "this is", "the result", "confirmed", "verified"]
        lower = content.lower()
        for phrase in uncertainty_phrases:
            if phrase in lower:
                score -= 0.1
        for phrase in confident_phrases:
            if phrase in lower:
                score += 0.05
        return max(0.1, min(1.0, score))

    def stream_think(self, prompt: str, system: Optional[str] = None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
