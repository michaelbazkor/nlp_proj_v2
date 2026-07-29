"""Task prompts for representation extraction."""
from __future__ import annotations


def build_user_prompt(task_prompt: str, corpus: str) -> str:
    return f"{task_prompt.strip()}\n\n--- USER POSTS ---\n{corpus.strip()}\n--- END ---\n"


def messages_for_chat(task_prompt: str, corpus: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": build_user_prompt(task_prompt, corpus),
        }
    ]
