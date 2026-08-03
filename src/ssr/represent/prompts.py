"""Task prompts for representation extraction."""
from __future__ import annotations

POSTS_PLACEHOLDER = "[insert posts here]"


def build_user_prompt(task_prompt: str, corpus: str) -> str:
    """Insert corpus into the task prompt.

    If ``task_prompt`` contains ``[insert posts here]``, replace that marker
    with the corpus. Otherwise append the corpus under legacy markers.
    """
    corpus = (corpus or "").strip() or "(no posts)"
    prompt = task_prompt.strip()
    if POSTS_PLACEHOLDER in prompt:
        return prompt.replace(POSTS_PLACEHOLDER, corpus, 1)
    return f"{prompt}\n\n--- USER POSTS ---\n{corpus}\n--- END ---\n"


def messages_for_chat(task_prompt: str, corpus: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": build_user_prompt(task_prompt, corpus),
        }
    ]
