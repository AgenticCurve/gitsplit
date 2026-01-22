"""AI interface for gitsplit using OpenRouter."""

import json
import os
from dataclasses import dataclass
from enum import Enum

import httpx


class ModelTier(str, Enum):
    """AI model tiers for escalation."""

    TIER1_FAST = "tier1"  # Fast, cheap model
    TIER2_REASONING = "tier2"  # Reasoning model
    TIER3_INTERACTIVE = "tier3"  # Claude Code / interactive


# Model configurations for each tier
TIER_MODELS = {
    ModelTier.TIER1_FAST: "anthropic/claude-sonnet-4",
    ModelTier.TIER2_REASONING: "anthropic/claude-sonnet-4",
    ModelTier.TIER3_INTERACTIVE: "anthropic/claude-sonnet-4",
}

# Approximate costs per 1M tokens (input/output)
MODEL_COSTS = {
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
}


@dataclass
class AIResponse:
    """Response from AI model."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    raw_response: dict | None = None


@dataclass
class TokenUsage:
    """Track token usage and costs."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    def add(self, response: AIResponse) -> None:
        """Add usage from a response."""
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost += response.cost


class AIError(Exception):
    """AI operation failed."""

    pass


class AIClient:
    """Client for AI operations via OpenRouter with conversation history."""

    def __init__(
        self,
        api_key: str | None = None,
        model_override: str | None = None,
        max_cost: float | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise AIError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key."
            )

        self.model_override = model_override
        self.max_cost = max_cost
        self.usage = TokenUsage()
        self.current_tier = ModelTier.TIER1_FAST

        self.base_url = "https://openrouter.ai/api/v1"
        self.client = httpx.Client(timeout=120.0)

        # Conversation history for multi-turn interactions
        self._conversation_history: list[dict[str, str]] = []
        self._current_system: str | None = None

    def reset_conversation(self, system: str | None = None) -> None:
        """Reset conversation history, optionally setting a new system prompt."""
        self._conversation_history = []
        self._current_system = system

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        self._conversation_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history."""
        self._conversation_history.append({"role": "assistant", "content": content})

    def add_error_context(self, error: str, diagnosis: str | None = None) -> None:
        """Add error context for self-healing retry."""
        error_msg = f"The previous attempt failed with this error:\n{error}"
        if diagnosis:
            error_msg += f"\n\nDiagnosis:\n{diagnosis}"
        error_msg += "\n\nPlease analyze what went wrong and provide a corrected response."
        self.add_user_message(error_msg)

    def get_conversation_length(self) -> int:
        """Get the number of messages in conversation history."""
        return len(self._conversation_history)

    def _get_model(self) -> str:
        """Get the model to use for the current request."""
        if self.model_override:
            return self.model_override
        return TIER_MODELS[self.current_tier]

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        costs = MODEL_COSTS.get(model, {"input": 3.0, "output": 15.0})
        return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000

    def _check_budget(self, estimated_cost: float) -> None:
        """Check if request would exceed budget."""
        if self.max_cost is not None:
            if self.usage.total_cost + estimated_cost > self.max_cost:
                raise AIError(
                    f"Request would exceed budget. "
                    f"Current: ${self.usage.total_cost:.4f}, "
                    f"Estimated: ${estimated_cost:.4f}, "
                    f"Max: ${self.max_cost:.2f}"
                )

    def escalate_tier(self) -> bool:
        """Escalate to next tier. Returns False if already at max."""
        if self.model_override:
            return False

        if self.current_tier == ModelTier.TIER1_FAST:
            self.current_tier = ModelTier.TIER2_REASONING
            return True
        elif self.current_tier == ModelTier.TIER2_REASONING:
            self.current_tier = ModelTier.TIER3_INTERACTIVE
            return True
        return False

    def reset_tier(self) -> None:
        """Reset to tier 1."""
        if not self.model_override:
            self.current_tier = ModelTier.TIER1_FAST

    def complete(
        self,
        messages: list[dict[str, str]] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_conversation: bool = False,
    ) -> AIResponse:
        """
        Complete a chat conversation.

        Args:
            messages: Messages to send. If use_conversation=True, these are appended to history.
            system: System prompt. If use_conversation=True and None, uses stored system.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            use_conversation: If True, use and update conversation history.

        Returns:
            AIResponse with the model's response.
        """
        model = self._get_model()

        # Determine which messages to send
        if use_conversation:
            # Add new messages to history
            if messages:
                for msg in messages:
                    self._conversation_history.append(msg)
            request_messages = list(self._conversation_history)
            system = system or self._current_system
        else:
            request_messages = messages or []

        # Rough token estimation for budget check
        text_len = sum(len(m.get("content", "")) for m in request_messages)
        if system:
            text_len += len(system)
        estimated_input = text_len // 4  # rough approximation
        estimated_output = max_tokens // 2
        estimated_cost = self._estimate_cost(model, estimated_input, estimated_output)

        self._check_budget(estimated_cost)

        # Build final message list with system prompt
        final_messages = []
        if system:
            final_messages.append({"role": "system", "content": system})
        final_messages.extend(request_messages)

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/gitsplit",
                    "X-Title": "gitsplit",
                },
                json={
                    "model": model,
                    "messages": final_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )

            if response.status_code != 200:
                raise AIError(f"API request failed: {response.status_code} - {response.text}")

            data = response.json()

            # Extract usage
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            actual_cost = self._estimate_cost(model, input_tokens, output_tokens)

            # Extract content
            content = ""
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content", "")

            result = AIResponse(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=actual_cost,
                raw_response=data,
            )

            self.usage.add(result)

            # Add assistant response to conversation history if using conversation
            if use_conversation:
                self._conversation_history.append({"role": "assistant", "content": content})

            return result

        except httpx.RequestError as e:
            raise AIError(f"Network error: {e}")

    def close(self) -> None:
        """Close the client."""
        self.client.close()


# Prompt templates for different phases

INTENT_DISCOVERY_SYSTEM = """You are an expert code analyst. Your task is to analyze git diffs and identify distinct logical intents (separate tasks/changes) within the code changes.

For each intent, identify:
1. A short, descriptive name (e.g., "Refactor database pooling", "Fix header alignment")
2. A brief description of what the intent accomplishes
3. Which files and line ranges belong to this intent

CRITICAL - Line Number Rules:
- Line ranges MUST refer to the NEW file (feature branch), not the old file
- For additions (+ lines), use the line number in the RESULTING file after the change
- Read the hunk header carefully: @@ -old_start,old_count +new_start,new_count @@
  Example: @@ -3,3 +3,8 @@ means old file has 3 lines starting at line 3,
  new file has 8 lines starting at line 3 (so new file ends at line 10)
- If lines 6-10 are added to a file, specify line_ranges: [[6, 10]]
- Be PRECISE with line numbers - verification will fail if ranges are wrong
- NEVER split a function/class definition across intents - include the ENTIRE function body
- If a function spans lines 26-35, include ALL of lines 26-35 - do not cut off early
- Count the actual lines in the diff output to get exact ranges
- If a function has changes related to different intents, assign the ENTIRE function to ONE intent

Guidelines:
- Look for semantic boundaries - changes that serve different purposes
- A single file may contain changes for multiple intents
- Consider dependencies between changes
- Be conservative - when in doubt, keep related changes together
- For files where ALL changes belong to one intent, set is_entire_file: true

Output your analysis as JSON with this structure:
{
  "intents": [
    {
      "id": "intent-a",
      "name": "Short descriptive name",
      "description": "What this intent accomplishes",
      "files": [
        {
          "path": "path/to/file.py",
          "line_ranges": [[start, end], [start2, end2]],
          "is_entire_file": false
        }
      ]
    }
  ],
  "reasoning": "Brief explanation of why you grouped changes this way"
}"""

CHANGE_PLANNING_SYSTEM = """You are an expert at surgical code splitting. Given a list of confirmed intents and the full diff, create a precise plan mapping every changed line to its intent.

For each file, specify exactly which lines belong to which intent. Handle edge cases:
- Lines shared by multiple intents: mark as "shared" with resolution strategy
- Dependencies: if Intent B's changes reference Intent A's changes, note the dependency
- Adjacent/overlapping lines: carefully determine boundaries

Output as JSON:
{
  "file_plans": [
    {
      "path": "file.py",
      "assignments": [
        {"lines": [10, 20], "intent_id": "intent-a"},
        {"lines": [21, 25], "intent_id": "intent-b"},
        {"lines": [26, 30], "intent_id": "shared", "shared_by": ["intent-a", "intent-b"], "strategy": "stack"}
      ]
    }
  ],
  "dependencies": [
    {"from": "intent-b", "to": "intent-a", "reason": "B uses function added in A"}
  ],
  "execution_order": ["intent-a", "intent-b", "intent-c"]
}"""


def parse_json_response(content: str) -> dict:
    """Parse JSON from AI response, handling markdown code blocks and preamble text."""
    content = content.strip()

    # Try to find JSON in markdown code block first
    if "```json" in content or "```" in content:
        import re
        # Match ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()

    # If content doesn't start with {, try to find JSON object
    if not content.startswith("{"):
        # Find first { and last }
        start_idx = content.find("{")
        if start_idx != -1:
            # Find the matching closing brace
            brace_count = 0
            end_idx = -1
            for i, char in enumerate(content[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if end_idx > start_idx:
                content = content[start_idx:end_idx]

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise AIError(f"Failed to parse AI response as JSON: {e}\nContent: {content[:500]}")
