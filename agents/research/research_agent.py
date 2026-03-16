"""
Research Agent — Deep analysis, knowledge synthesis, information gathering
"""
import httpx
import json
from typing import Dict, Any
from loguru import logger
from agents.base_agent import BaseAgent, AgentResult
from core.memory.memory import MemorySystem

class ResearchAgent(BaseAgent):
    agent_type = "research"

    SYSTEM = """You are a world-class research analyst and knowledge synthesizer.
You break down complex topics into clear, structured insights.
You cite reasoning, identify patterns, and build comprehensive understanding.
You are precise, thorough, and always grounded in facts.
You distinguish between what is known, what is uncertain, and what requires more investigation."""

    def __init__(self, memory=None):
        super().__init__(memory=memory, model_tier="powerful")

    def execute(self, task: str, context: Dict = None) -> AgentResult:
        ctx_str = ""
        if context:
            ctx_str = f"\nAdditional context: {json.dumps(context, indent=2)}"
        
        prompt = f"""Research Task: {task}{ctx_str}

Provide a comprehensive, structured analysis including:
1. Key findings
2. Important details and nuances
3. Relevant considerations
4. Actionable insights or recommendations
5. Confidence level and any uncertainties

Be thorough but organized."""
        
        return self._run(prompt, self.SYSTEM, task, context)

    def analyze(self, topic: str, depth: str = "comprehensive") -> AgentResult:
        """Deep analysis of a topic"""
        prompt = f"""Analyze the following at {depth} depth:

{topic}

Structure your analysis:
## Overview
## Core Components
## Key Relationships  
## Critical Insights
## Recommendations
## Confidence Assessment"""
        return self._run(prompt, self.SYSTEM, f"analyze: {topic[:50]}")

    def compare(self, items: list, criteria: list = None) -> AgentResult:
        """Compare multiple items across criteria"""
        criteria_str = ", ".join(criteria) if criteria else "all relevant dimensions"
        items_str = "\n".join([f"- {item}" for item in items])
        
        prompt = f"""Compare these items across {criteria_str}:

{items_str}

Provide:
1. Structured comparison table (text format)
2. Key differentiators
3. Recommendation with justification"""
        return self._run(prompt, self.SYSTEM, "comparison analysis")

    def summarize(self, text: str, format: str = "bullets") -> AgentResult:
        """Summarize content"""
        prompt = f"""Summarize the following content in {format} format:

{text[:4000]}

Focus on: key points, important details, actionable items."""
        return self._run(prompt, self.SYSTEM, "summarization")
