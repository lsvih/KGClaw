"""
Agent system for KGClaw.

Implements LLM-powered agents that can:
- Use tools (function calling)
- Spawn subagents for parallel work
- Produce structured output
- Maintain conversation context via memory
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from openai import OpenAI
from openai import AuthenticationError

from .logger import get_logger

from .memory import Memory
from .models import (
    LLMConfig,
    Message,
    Role,
    ToolDefinition,
)
from .tools import discover_tools, execute_tool


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    name: str = "agent"
    system_prompt: str = ""
    model_config: Optional[LLMConfig] = None
    tools: list[str] = field(default_factory=list)  # tool names this agent can use
    parent_id: Optional[str] = None
    max_tool_calls: int = 20
    structured_output_schema: Optional[dict[str, Any]] = None


class Agent:
    """
    An LLM-powered agent that can use tools and spawn subagents.

    Each agent has:
    - A system prompt defining its role and behavior
    - Access to a set of tools (function calling)
    - Conversation memory (through the shared Memory object)
    - The ability to spawn child subagents
    """

    def __init__(
        self,
        config: AgentConfig,
        memory: Memory,
        llm_config: LLMConfig,
    ):
        self.config = config
        self.memory = memory
        self.llm_config = config.model_config or llm_config  # Agent-level overrides harness
        self.agent_id = config.name

        self._client: Optional[OpenAI] = None
        self._messages: list[Message] = []
        self._callbacks: list[Callable[[str, dict], None]] = []
        self._log = get_logger()
        # Token tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.llm_config.api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
                base_url=self.llm_config.api_base,
                max_retries=3,  # Built-in retry with exponential backoff
            )
        return self._client

    def on_event(self, callback: Callable[[str, dict], None]):
        """Register a callback for agent events (tool_calls, subagent_start, etc.)."""
        self._callbacks.append(callback)

    def _emit(self, event_type: str, data: dict[str, Any]):
        for cb in self._callbacks:
            try:
                cb(event_type, data)
            except Exception:
                pass

    def add_message(self, role: Role, content: str, **kwargs):
        """Add a message to this agent's conversation."""
        msg = Message(role=role, content=content, **kwargs)
        self._messages.append(msg)

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for this agent."""
        return discover_tools(self.config.tools)

    def run(
        self,
        user_message: str,
        max_iterations: int = 10,
    ) -> str:
        """
        Run the agent with a user message and return the final response.

        The agent will:
        1. Send the message to the LLM
        2. If the LLM calls tools, execute them and send results back
        3. Repeat until the LLM responds without tool calls or max iterations

        Args:
            user_message: The task/message to send to the agent
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            The agent's final text response
        """
        # Build initial messages
        messages: list[dict[str, Any]] = []

        # System prompt
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        # Load existing messages from memory (auto-compact if too many)
        self.memory.compact_messages(self.agent_id, max_messages=50)
        for msg in self.memory.get_messages(self.agent_id):
            messages.append(msg.model_dump(exclude_none=True))

        # Add user message
        messages.append({"role": "user", "content": user_message})
        self.memory.add_message(self.agent_id, Message(role=Role.USER, content=user_message))

        # Tool definitions
        tools = self._get_tool_definitions()
        tool_schemas = [t.to_openai_format() for t in tools] if tools else None

        # Main loop
        tools_exhausted = False          # hard-stop after max_tool_calls
        consecutive_failures = 0         # circuit breaker for tool failures
        MAX_CONSECUTIVE_FAILURES = 3     # after 3 consecutive failed tool calls, force stop

        for iteration in range(max_iterations):
            kwargs: dict[str, Any] = {
                "model": self.llm_config.model,
                "messages": messages,
                "temperature": self.llm_config.temperature,
                "max_tokens": self.llm_config.max_tokens,
            }

            # After max_tool_calls are exhausted, strip tools so the LLM is
            # forced to produce a text response instead of looping on tool calls.
            if tools_exhausted:
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
            elif tool_schemas:
                kwargs["tools"] = tool_schemas
                kwargs["tool_choice"] = "auto"

            if self.config.structured_output_schema and not tool_schemas:
                # Use JSON mode for structured output when no tools
                kwargs["response_format"] = {"type": "json_object"}

            try:
                # Trace: capture full LLM request before sending
                self._log.trace_llm_request(
                    agent=self.agent_id,
                    model=self.llm_config.model,
                    prompt=_messages_to_text(messages),
                    tools=tool_schemas,
                )

                response = self.client.chat.completions.create(**kwargs)
            except AuthenticationError:
                raise RuntimeError(
                    "LLM API 密钥无效或已过期。请运行 'kgclaw setup' 重新配置，"
                    "或设置环境变量 OPENAI_API_KEY。"
                )
            choice = response.choices[0]
            assistant_message = choice.message

            # Track token usage — always emit even if API doesn't return usage
            if response.usage:
                self.prompt_tokens += response.usage.prompt_tokens
                self.completion_tokens += response.usage.completion_tokens
            else:
                # Fallback: approximate from character counts (4 chars ≈ 1 token)
                prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
                completion_chars = len(assistant_message.content or "")
                self.prompt_tokens += prompt_chars // 4
                self.completion_tokens += completion_chars // 4
            self._emit("token_usage", {
                "_agent_id": self.agent_id,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            })

            # Trace: capture full LLM response
            self._log.trace_llm_response(
                agent=self.agent_id,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                content=assistant_message.content or "",
                tool_calls=[
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in (assistant_message.tool_calls or [])
                ] if assistant_message.tool_calls else None,
                finish_reason=choice.finish_reason or "",
            )

            # Check for tool calls
            if assistant_message.tool_calls and not tools_exhausted:
                # Add assistant message with tool calls
                tool_calls_data = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in assistant_message.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": tool_calls_data,
                })

                # Execute tools
                any_success = False
                for tc in assistant_message.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    self._emit("tool_call", {
                        "agent": self.agent_id,
                        "tool": tool_name,
                        "args": tool_args,
                    })

                    self._log.tool_call(self.agent_id, tool_name, tool_args)
                    result = execute_tool(tool_name, tool_args)
                    self._log.tool_result(
                        self.agent_id, tool_name, result.success,
                        str(result.data)[:200] if result.success else str(result.error),
                    )

                    # Add tool result
                    result_content = json.dumps(result.data, ensure_ascii=False) if result.success else f"Error: {result.error}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_content,
                    })

                    self._emit("tool_result", {
                        "agent": self.agent_id,
                        "tool": tool_name,
                        "success": result.success,
                    })

                    if result.success:
                        any_success = True
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                # Circuit breaker: too many consecutive tool failures → force text response
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self._log.warning(
                        f"Agent {self.agent_id}: {consecutive_failures} consecutive tool failures, "
                        f"forcing text response (circuit breaker)"
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            f"连续 {consecutive_failures} 次工具调用失败了。"
                            "请不要再调用工具，直接基于已有信息给出最终回答。"
                        ),
                    })
                    tools_exhausted = True

                # Hard-stop after max_tool_calls: strip tools, force text response
                if iteration >= self.config.max_tool_calls - 1:
                    self._log.info(
                        f"Agent {self.agent_id}: max_tool_calls ({self.config.max_tool_calls}) "
                        f"reached at iteration {iteration+1}, forcing text response"
                    )
                    messages.append({
                        "role": "user",
                        "content": "已达到最大工具调用次数。请不要再调用任何工具，直接基于当前信息给出最终回答。",
                    })
                    tools_exhausted = True

            else:
                # No tool calls — final response
                content = assistant_message.content or ""
                messages.append({"role": "assistant", "content": content})
                self.memory.add_message(self.agent_id, Message(role=Role.ASSISTANT, content=content))
                return content

        # Max iterations reached
        return "Agent stopped: maximum iterations reached without final response."

    def run_stream(
        self,
        user_message: str,
        max_iterations: int = 10,
    ):
        """
        Run the agent with streaming output.

        Yields tuples of (event_type, data) where event_type is:
        - 'token': a text token from the LLM response
        - 'tool_call': the agent is calling a tool
        - 'tool_result': a tool execution result
        - 'thinking': the agent is starting to think (iteration start)
        - 'done': final response complete
        - 'error': an error occurred

        Args:
            user_message: The task/message to send to the agent
            max_iterations: Maximum number of tool-calling iterations

        Yields:
            (str, Any) tuples representing streaming events
        """
        messages: list[dict[str, Any]] = []

        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        # Load and auto-compact context from previous interactions
        self.memory.compact_messages(self.agent_id, max_messages=50)
        for msg in self.memory.get_messages(self.agent_id):
            messages.append(msg.model_dump(exclude_none=True))

        messages.append({"role": "user", "content": user_message})
        self.memory.add_message(self.agent_id, Message(role=Role.USER, content=user_message))

        tools = self._get_tool_definitions()
        tool_schemas = [t.to_openai_format() for t in tools] if tools else None

        tools_exhausted = False
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 3

        for iteration in range(max_iterations):
            yield ("thinking", {"iteration": iteration + 1, "agent": self.agent_id})

            # After max_tool_calls are exhausted, strip tools to force text response
            stream_tools = None if tools_exhausted else tool_schemas
            stream_tool_choice = None if tools_exhausted else ("auto" if tool_schemas else None)

            try:
                stream = self.client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=messages,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens,
                    tools=stream_tools,
                    tool_choice=stream_tool_choice,
                    stream=True,
                )
            except AuthenticationError:
                yield ("error", {"message": "API 密钥无效。请运行 'kgclaw setup' 重新配置，或设置环境变量 OPENAI_API_KEY。"})
                return
            except Exception as e:
                yield ("error", {"message": str(e)})
                return

            # Accumulate streaming response
            assistant_content = ""
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            finish_reason = None

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

                if delta is None:
                    continue

                # Text tokens
                if delta.content:
                    assistant_content += delta.content
                    yield ("token", {"text": delta.content, "agent": self.agent_id})

                # Tool call tokens (streaming tool calls arrive in pieces)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.id:
                            tool_calls_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

            # Track token usage from streaming (approximate: count chars)
            # Streaming doesn't return usage until final chunk in some APIs
            if finish_reason:
                approx_prompt = sum(len(str(m.get("content", ""))) for m in messages) // 4
                approx_completion = len(assistant_content) // 4
                self.prompt_tokens += approx_prompt
                self.completion_tokens += approx_completion
                self._emit("token_usage", {
                    "_agent_id": self.agent_id,
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                })

            # Handle tool calls (if any were accumulated)
            if tool_calls_acc and not tools_exhausted:
                tool_calls_data = []
                tool_results_map: dict[int, str] = {}  # idx -> result_content for reuse
                any_success = False
                for idx in sorted(tool_calls_acc.keys()):
                    tc = tool_calls_acc[idx]
                    tool_name = tc["function"]["name"]
                    try:
                        tool_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}

                    self._log.tool_call(self.agent_id, tool_name, tool_args)

                    yield ("tool_call", {
                        "agent": self.agent_id,
                        "tool": tool_name,
                        "args": tool_args,
                    })

                    # Execute tool ONCE — store result for both the stream event
                    # and the LLM context message below.
                    result = execute_tool(tool_name, tool_args)
                    self._log.tool_result(
                        self.agent_id, tool_name, result.success,
                        str(result.data)[:200] if result.success else str(result.error),
                    )
                    result_content = json.dumps(result.data, ensure_ascii=False) if result.success else f"Error: {result.error}"
                    tool_results_map[idx] = result_content

                    yield ("tool_result", {
                        "agent": self.agent_id,
                        "tool": tool_name,
                        "success": result.success,
                        "result": result_content[:500],
                    })

                    if result.success:
                        any_success = True
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                    tool_calls_data.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    })

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": assistant_content or "",
                    "tool_calls": tool_calls_data,
                })

                # Add tool results (reuse stored results — no re-execution)
                for idx in sorted(tool_calls_acc.keys()):
                    tc = tool_calls_acc[idx]
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_results_map.get(idx, "Error: tool result unavailable"),
                    })

                # Circuit breaker: too many consecutive tool failures → force text response
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self._log.warning(
                        f"Agent {self.agent_id}: {consecutive_failures} consecutive tool failures, "
                        f"forcing text response (circuit breaker)"
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            f"连续 {consecutive_failures} 次工具调用失败了。"
                            "请不要再调用工具，直接基于已有信息给出最终回答。"
                        ),
                    })
                    tools_exhausted = True

                # Hard-stop after max_tool_calls: strip tools, force text response
                if iteration >= self.config.max_tool_calls - 1:
                    self._log.info(
                        f"Agent {self.agent_id}: max_tool_calls ({self.config.max_tool_calls}) "
                        f"reached at iteration {iteration+1}, forcing text response"
                    )
                    messages.append({
                        "role": "user",
                        "content": "已达到最大工具调用次数。请不要再调用任何工具，直接基于当前信息给出最终回答。",
                    })
                    tools_exhausted = True

            else:
                # No tool calls — this is the final response
                messages.append({"role": "assistant", "content": assistant_content})
                self.memory.add_message(
                    self.agent_id,
                    Message(role=Role.ASSISTANT, content=assistant_content),
                )
                yield ("done", {"content": assistant_content, "agent": self.agent_id})
                return

        yield ("done", {"content": assistant_content, "agent": self.agent_id})

    def run_structured(
        self,
        user_message: str,
        output_schema: dict[str, Any],
        max_iterations: int = 10,
    ) -> Optional[dict[str, Any]]:
        """
        Run the agent and parse structured JSON output.

        Uses a two-step approach:
        1. Have the LLM produce JSON (with schema in the system prompt)
        2. Parse and validate the JSON

        Returns the parsed result or None on failure.
        """
        # Inject schema requirement into the user message
        schema_instruction = f"""
请严格按照以下 JSON Schema 格式输出结果。你的整个回复必须是一个合法的 JSON 对象，不要包含任何 JSON 之外的文字。

```json
{json.dumps(output_schema, indent=2, ensure_ascii=False)}
```

任务：
{user_message}
"""
        response = self.run(schema_instruction, max_iterations=max_iterations)

        # Parse JSON from response
        try:
            # Try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(pattern, response)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text
        brace_start = response.find("{")
        brace_end = response.rfind("}")
        bracket_start = response.find("[")
        bracket_end = response.rfind("]")

        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(response[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        if bracket_start != -1 and bracket_end > bracket_start:
            try:
                return json.loads(response[bracket_start:bracket_end + 1])
            except json.JSONDecodeError:
                pass

        # Tier 5: LLM-based JSON repair — ask LLM to fix malformed response
        try:
            return self._repair_json_via_llm(response, output_schema)
        except Exception:
            pass

        return None

    def _repair_json_via_llm(
        self,
        malformed_response: str,
        output_schema: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Send the malformed JSON response back to the LLM for repair.

        Asks the LLM to fix any syntax errors and return ONLY valid JSON.
        """
        repair_prompt = f"""Your previous response was not valid JSON. Please fix it and return ONLY a valid JSON object that matches the required schema. Do NOT include any text outside the JSON, no markdown code blocks, no explanation.

## Required JSON Schema
```json
{json.dumps(output_schema, indent=2, ensure_ascii=False)}
```

## Your Previous (malformed) Response
```
{malformed_response[:8000]}
```

## Instructions
1. Fix any JSON syntax errors (missing quotes, commas, braces, etc.)
2. Ensure the output strictly conforms to the schema above
3. Return ONLY the corrected JSON — no markdown, no explanation"""

        # Make a single repair call (no tools, low temperature for reliability)
        repair_resp = self.client.chat.completions.create(
            model=self.llm_config.model,
            messages=[{"role": "user", "content": repair_prompt}],
            temperature=0.1,
            max_tokens=self.llm_config.max_tokens,
        )
        raw = repair_resp.choices[0].message.content or ""

        # Track token usage from repair call
        if repair_resp.usage:
            self.prompt_tokens += repair_resp.usage.prompt_tokens
            self.completion_tokens += repair_resp.usage.completion_tokens

        # Parse the repaired response
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try same fallback strategies on the repaired response
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: try to get brace/brace content
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(raw[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def spawn_subagent(
        self,
        name: str,
        system_prompt: str,
        task: str,
        tools: Optional[list[str]] = None,
    ) -> str:
        """
        Spawn a subagent to handle a subtask.

        The subagent inherits this agent's LLM config and memory,
        but has its own conversation context and tool set.

        Args:
            name: Subagent identifier
            system_prompt: System prompt for the subagent
            task: The task to delegate
            tools: Tool names the subagent can use

        Returns:
            The subagent's final response
        """
        self._emit("subagent_start", {
            "parent": self.agent_id,
            "subagent": name,
            "task_preview": task[:100],
        })

        sub_config = AgentConfig(
            name=f"{self.agent_id}.{name}",
            system_prompt=system_prompt,
            model_config=self.config.model_config,
            tools=tools or [],
            parent_id=self.agent_id,
            max_tool_calls=self.config.max_tool_calls,
        )

        subagent = Agent(sub_config, self.memory, self.llm_config)

        # Forward events from subagent
        subagent.on_event(lambda et, d: self._emit(et, d))

        result = subagent.run(task)
        self._emit("subagent_done", {
            "parent": self.agent_id,
            "subagent": name,
            "result_preview": result[:200],
        })

        return result


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Convert a message list to a single text blob for trace logging.

    Formats each message as: [role] content (truncated at 8K chars per message).
    """
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = str(m.get("content", ""))
        if len(content) > 8192:
            content = content[:8192] + "\n...[truncated]..."
        tool_calls = m.get("tool_calls")
        if tool_calls:
            tc_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            content += f"\n[tool_calls: {', '.join(tc_names)}]"
        lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


def create_agent(
    config: AgentConfig,
    memory: Memory,
    llm_config: LLMConfig,
) -> Agent:
    """Factory function to create an agent instance."""
    return Agent(config, memory, llm_config)


def create_subagent_factory(parent: Agent):
    """
    Create a convenience function for spawning subagents from a parent agent.
    Returns a function: spawn(name, system_prompt, task, tools=None) -> str
    """
    return lambda name, system_prompt, task, tools=None: parent.spawn_subagent(
        name=name,
        system_prompt=system_prompt,
        task=task,
        tools=tools,
    )
