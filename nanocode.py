
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

class GrepTool(Tool):
    """Search files under a directory for lines matching a regex."""
    
    name = "grep"
    description = "Search files under a directory for lines matching a regex."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex to search for."},
            "path": {"type": "string", "description": "Directory to search in.", "default": "."},
        },
        "required": ["pattern"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            regex = re.compile(args["pattern"])
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches = []
        search_path = args.get("path", ".")
        
        for dirpath, _, filenames in os.walk(search_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, encoding="utf-8") as f:
                        for lineno, line in enumerate(f, 1):
                            if regex.search(line):
                                matches.append(f"{filepath}:{lineno}: {line.rstrip()}")
                except (UnicodeDecodeError, OSError):
                    continue
        
        return "\n".join(matches) if matches else "No matches found."


class BashTool(Tool):
    """Run a shell command."""
    
    name = "bash"
    description = "Run a shell command and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
        },
        "required": ["command"],
    }
    is_read_only = False

    def execute(self, args: dict[str, Any]) -> str:
        try:
            result = subprocess.run(
                args["command"], 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=30
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            return f"Error running command: {e}"


class TodoWriteTool(Tool):
    """Manage task lists."""
    
    name = "todo_write"
    description = "Write the current task list. Replaces the whole list each call."
    parameters = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["items"],
    }
    is_read_only = True

    def __init__(self):
        self.items: list[dict[str, str]] = []

    def execute(self, args: dict[str, Any]) -> str:
        self.items = args["items"]
        marks = {"pending": " ", "in_progress": "~", "done": "x"}
        return "\n".join(
            f"[{marks[item['status']]}] {item['content']}" 
            for item in self.items
        )


class WebFetchTool(Tool):
    """Fetch a URL and return its content."""
    
    name = "web_fetch"
    description = "Fetch a URL and return its content as readable text."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch."},
        },
        "required": ["url"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            response = requests.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                json={"url": args["url"], "formats": ["markdown"]},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["data"]["markdown"][:MAX_WEB_CONTENT_LENGTH]
        except requests.RequestException as e:
            return f"Error fetching URL: {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing response: {e}"


class WebSearchTool(Tool):
    """Search the web for information."""
    
    name = "web_search"
    description = "Search the web and return the top results (title, URL, description)."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
        },
        "required": ["query"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            response = requests.post(
                "https://api.firecrawl.dev/v2/search",
                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                json={"query": args["query"], "limit": 5, "sources": ["web"]},
                timeout=30
            )
            response.raise_for_status()
            results = response.json()["data"]["web"]
            return "\n\n".join(
                f"{r['title']}\n{r['url']}\n{r.get('description', '')}" 
                for r in results
            )
        except requests.RequestException as e:
            return f"Error searching web: {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing search results: {e}"
