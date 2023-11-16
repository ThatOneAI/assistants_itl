import asyncio
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
import os
import json

from transformers.tools import Agent, Tool, OpenAiAgent
from itllib import Itl
from itllib.clusters import ResourceController

from .metatools import _check_config


class HFMetaAssistantModule:
    def __init__(self, itl: Itl, cluster, stream, agent, prompts, examples, tools):
        self.itl = itl
        self.cluster = cluster
        self.stream = stream
        self.agent: Agent = agent

        self.prompts = prompts
        self.examples = examples

        self.tools = tools
        self.tools_changed = True
        self.current_tools = {}
        self.current_examples = []
        self.current_mission = None

        itl.ondata(stream)(self._handle_messages)

    def configure(self, spec: dict):
        required_keys = {"tools", "stream"}
        optional_keys = {"examples", "mission"}
        _check_config(spec, required_keys, optional_keys)

        self.current_tools = spec["tools"].copy()
        self.tools_changed = True
        self.current_examples = spec.get("examples", [])
        self.current_mission = spec.get("mission", None)

    def run(self, message: str):
        self.agent.prepare_for_new_chat()
        return self.chat(message)

    def chat(self, message: str):
        print("Incoming message:", message)
        if self.agent.cached_tools:
            self.agent.cached_tools.clear()
        for name, reference in self.current_tools.items():
            if reference in self.tools:
                self.agent.toolbox[name] = self.tools[reference]
        tools = {name: tool.description for name, tool in self.agent.toolbox.items()}
        self.agent.chat_history = self._assemble_prompt(tools)
        self.tools_changed = False

        print("=== Result ===")
        result = self.agent.chat(message)
        print(result)

        return result

    def _assemble_tool_description(self, tools):
        tool_lines = []
        for name, description in tools.items():
            tool_lines.append(f"- {name}: {description}")
        tool_description = "\n".join(tool_lines)
        return f"Tools:\n{tool_description}"

    def _assemble_task_log(self, tasklog, tools={}):
        parts = []

        if tools:
            parts.append(self._assemble_tool_description(tools))
            parts.append("=====")

        parts.append(f"Human: {tasklog['prompt']}")
        parts.append(f"Assistant: {tasklog['steps']}")
        parts.append(f"```python\n{tasklog['code']}\n```")

        return "\n\n".join(parts)

    def _assemble_prompt(self, current_tools):
        tasks = []

        mission = self.prompts.get(self.current_mission, None)
        if mission:
            tasks.append(mission)

        previous_tools = {}

        for task_name in self.current_examples:
            if task_name not in self.examples:
                print(f"Missing example task: {task_name}")
                continue

            task = self.examples[task_name]
            if previous_tools != task.get("tools", {}):
                tools = task.get("tools", {})
                previous_tools = tools
            else:
                tools = {}
            tasks.append(self._assemble_task_log(task, tools))

        if current_tools != previous_tools:
            tasks.append(self._assemble_tool_description(current_tools))

        return "\n\n".join(tasks) + "\n"

    async def _handle_messages(self, message):
        if not isinstance(message, str):
            print(f"Invalid message type: {type(message)}")
            return

        if message.startswith(">"):
            return None

        if message.startswith("+"):
            mode = "chat"
            message = message[1:]
        else:
            mode = "run"

        if not isinstance(message, str):
            print(f"Invalid message type: {type(message)}")
            return

        message = message.strip()
        if len(message) == 0:
            print("Rejecting empty message")
            return

        if mode == "run":
            response = self.run(message)
        elif mode == "chat":
            response = self.chat(message)
        else:
            print(f"Invalid mode: {mode}. Expected 'chat' or 'run'")
            return

        try:
            asyncio.create_task(self.itl.stream_send(self.stream, f">{response}"))
            print("Done processing message")
        except Exception as e:
            print("Failed to send response:", e)
