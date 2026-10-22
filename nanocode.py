
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
