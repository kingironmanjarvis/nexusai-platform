"""
Optimizer Agent — Performance analysis, improvement proposals, architecture upgrades
"""
import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentResult

class OptimizerAgent(BaseAgent):
    agent_type = "optimizer"

    SYSTEM = """You are a systems optimization expert.
You analyze performance metrics, identify bottlenecks, and propose concrete improvements.
You think in terms of: speed, efficiency, reliability, scalability, and cost.
You prioritize improvements by impact and feasibility.
You produce actionable optimization plans, not vague suggestions."""

    def __init__(self, memory=None):
        super().__init__(memory=memory, model_tier="balanced")

    def execute(self, task: str, context: Dict = None) -> AgentResult:
        metrics = context.get("metrics", {}) if context else {}
        system_info = context.get("system_info", "") if context else ""
        
        prompt = f"""Optimization Task: {task}

Current Metrics:
{json.dumps(metrics, indent=2) if metrics else "No metrics provided"}

System Info: {system_info}

Provide:
1. BOTTLENECKS: Top 3 performance issues
2. QUICK_WINS: Improvements achievable immediately
3. STRATEGIC: Longer-term architecture improvements
4. EXPECTED_GAINS: Quantified improvement estimates
5. PRIORITY_ORDER: Which to tackle first and why"""
        
        return self._run(prompt, self.SYSTEM, task, context)

    def propose_improvements(self, codebase_summary: str, metrics: Dict) -> AgentResult:
        prompt = f"""Analyze this system and propose improvements.

SYSTEM SUMMARY:
{codebase_summary[:2000]}

PERFORMANCE METRICS:
{json.dumps(metrics, indent=2)}

Propose:
1. Architecture improvements
2. Code optimizations
3. Infrastructure changes
4. Algorithm improvements
Rank by expected impact."""
        return self._run(prompt, self.SYSTEM, "propose improvements")

    def benchmark_comparison(self, before: Dict, after: Dict) -> AgentResult:
        prompt = f"""Compare these before/after performance metrics.

BEFORE:
{json.dumps(before, indent=2)}

AFTER:
{json.dumps(after, indent=2)}

Calculate:
1. % improvement per metric
2. Overall performance delta
3. Whether changes were beneficial
4. Any regressions detected"""
        return self._run(prompt, self.SYSTEM, "benchmark comparison")
