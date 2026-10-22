
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from openai import OpenAI

# Configuration
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "poolside/laguna-s-2.1:free"
FIRECRAWL_API_KEY = os.environ["FIRECRAWL_API_KEY"]
MAX_WEB_CONTENT_LENGTH = 5000

# Initialize client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: str = ""


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    auto_approve: bool = False
    plan_mode: bool = False
    max_iterations: int = 50


class Tool:
    """Base class for all tools."""
    
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    is_read_only: bool = False

    def execute(self, args: dict[str, Any]) -> str:
        """Execute the tool with given arguments."""
        raise NotImplementedError

    def to_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ReadFileTool(Tool):
    """Read a file from disk."""
    
    name = "read_file"
    description = "Read a file from disk and return its contents."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
        },
        "required": ["path"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            with open(args["path"], encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File not found: {args['path']}"
        except PermissionError:
            return f"Error: Permission denied: {args['path']}"
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFileTool(Tool):
    """Write content to a file."""
    
    name = "write_file"
    description = "Write content to a file, creating or overwriting it."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["path", "content"],
    }
    is_read_only = False

    def execute(self, args: dict[str, Any]) -> str:
        try:
            with open(args["path"], "w", encoding="utf-8") as f:
                f.write(args["content"])
            return f"Wrote {args['path']}"
        except Exception as e:
            return f"Error writing file: {e}"


class EditFileTool(Tool):
    """Replace an exact string in a file."""
    
    name = "edit_file"
    description = "Replace an exact string in a file with a new string."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit."},
            "old_string": {"type": "string", "description": "Exact string to replace."},
            "new_string": {"type": "string", "description": "Replacement string."},
        },
        "required": ["path", "old_string", "new_string"],
    }
    is_read_only = False

    def execute(self, args: dict[str, Any]) -> str:
        try:
            with open(args["path"], encoding="utf-8") as f:
                content = f.read()
            
            if args["old_string"] not in content:
                return f"Error: old_string not found in {args['path']}"
            
            updated_content = content.replace(args["old_string"], args["new_string"])
            with open(args["path"], "w", encoding="utf-8") as f:
                f.write(updated_content)
            return f"Edited {args['path']}"
        except FileNotFoundError:
            return f"Error: File not found: {args['path']}"
        except Exception as e:
            return f"Error editing file: {e}"

