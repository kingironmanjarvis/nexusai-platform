"""
Reviewer Agent — Quality assurance, validation, critique, improvement
"""
import json
from typing import Dict, Any
from agents.base_agent import BaseAgent, AgentResult

class ReviewerAgent(BaseAgent):
    agent_type = "reviewer"

    SYSTEM = """You are a senior quality assurance engineer and critical reviewer.
Your job is to find flaws, validate correctness, and ensure quality.
You are constructive but uncompromising about quality.
You provide specific, actionable feedback — never vague criticism.
You validate that outputs actually accomplish what was requested."""

    def __init__(self, memory=None):
        super().__init__(memory=memory, model_tier="balanced")

    def execute(self, task: str, context: Dict = None) -> AgentResult:
        content_to_review = context.get("content", "") if context else ""
        original_task = context.get("original_task", task) if context else task
        
        prompt = f"""Review this output for quality and correctness.

ORIGINAL TASK: {original_task}

OUTPUT TO REVIEW:
{content_to_review[:3000]}

Evaluate:
1. COMPLETENESS: Does it fully address the task? (0-10)
2. ACCURACY: Is it factually correct? (0-10)
3. QUALITY: Is it well-structured and clear? (0-10)
4. ISSUES: List specific problems found
5. VERDICT: PASS / NEEDS_REVISION / FAIL
6. IMPROVEMENTS: Specific changes needed

Return JSON."""
        
        return self._run(prompt, self.SYSTEM, task, context)

    def validate_code(self, code: str, requirements: str) -> AgentResult:
        prompt = f"""Validate this code against requirements.

REQUIREMENTS: {requirements}

CODE:
```
{code[:3000]}
```

Check:
1. Does it meet all requirements?
2. Are there bugs or edge cases?
3. Security issues?
4. Performance concerns?
5. VERDICT: PASS / FAIL with reasons"""
        return self._run(prompt, self.SYSTEM, "code validation")

    def score_output(self, output: str, task: str) -> Dict:
        """Quick scoring of an output"""
        result = self.execute(task, context={"content": output, "original_task": task})
        try:
            content = result.result
            # Try to parse JSON score
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except:
            pass
        return {"verdict": "unknown", "content": result.result}
