"""
Coding Agent — Writes, executes, debugs, and improves code
"""
import subprocess
import tempfile
import os
import re
from typing import Dict, Any, List
from loguru import logger
from agents.base_agent import BaseAgent, AgentResult
from core.memory.memory import MemorySystem

class CodingAgent(BaseAgent):
    agent_type = "coding"

    SYSTEM = """You are an elite software engineer and coding agent.
You write clean, working, production-ready code.
You NEVER write placeholder code or mockups.
When asked to write code, you write real, executable code that actually works.
You think through the problem before writing.
You handle edge cases, errors, and validation.
Always explain what the code does briefly after writing it."""

    def __init__(self, memory=None):
        super().__init__(memory=memory, model_tier="powerful")

    def execute(self, task: str, context: Dict = None) -> AgentResult:
        ctx_str = ""
        if context:
            ctx_str = f"\nContext: {context}"
        
        prompt = f"""Task: {task}{ctx_str}

Write complete, working code to accomplish this task.
If execution is needed, provide the code in a ```python block.
Include error handling and logging."""
        
        result = self._run(prompt, self.SYSTEM, task, context)
        
        # Try to execute any Python code in the result
        if result.success:
            code_blocks = self._extract_code(result.result)
            executed_outputs = []
            for lang, code in code_blocks:
                if lang in ("python", "py", ""):
                    output = self._execute_code(code)
                    executed_outputs.append(output)
                    if output["success"]:
                        logger.info(f"[CodingAgent] Code executed successfully")
                    else:
                        logger.warning(f"[CodingAgent] Code execution failed: {output.get('error', output.get('stderr', 'unknown error'))}")
            
            if executed_outputs:
                result.artifacts = executed_outputs
        
        return result

    def write_and_run(self, code: str, language: str = "python") -> Dict:
        """Execute arbitrary code in sandbox"""
        return self._execute_code(code)

    def debug(self, code: str, error: str) -> AgentResult:
        """Debug failing code"""
        prompt = f"""This code has an error. Fix it.

CODE:
```python
{code}
```

ERROR:
{error}

Provide the fixed code with explanation of what was wrong."""
        return self._run(prompt, self.SYSTEM, f"debug: {error[:50]}")

    def review(self, code: str) -> AgentResult:
        """Code review and improvement suggestions"""
        prompt = f"""Review this code for quality, bugs, performance, and security issues.

```
{code}
```

Provide:
1. Issues found (if any)
2. Severity (critical/warning/info)
3. Improved version of the code"""
        return self._run(prompt, self.SYSTEM, "code review")

    def _extract_code(self, text: str) -> List[tuple]:
        """Extract code blocks from markdown"""
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def _execute_code(self, code: str) -> Dict:
        """Execute Python code in isolated subprocess"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            tmp_path = f.name
        
        try:
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "PYTHONPATH": "/app/platform"}
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:1000],
                "returncode": result.returncode,
                "code": code[:500]
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Execution timed out (30s)", "code": code[:500]}
        except Exception as e:
            return {"success": False, "error": str(e), "code": code[:500]}
        finally:
            os.unlink(tmp_path)
