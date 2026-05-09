"""
Agent runner module.

Runs a question through an OpenAI model via LangChain and captures:
- The full response text for each conversation turn
- Token-level logprobs (probability + top alternatives per token)
- Token usage and finish reason metadata

The multi-turn flow is:
  Turn 1: Present the question as-is.
  Turn 2+: Ask a confidence follow-up to probe the model's certainty.

Logprobs are the core signal for the hallucination detection research:
when a model accepts a nonsensical premise, the logprob distribution
of its output tokens is expected to differ from cases where it pushes back.
"""

import os
from datetime import datetime
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

SYSTEM_PROMPT = (
    "You are a knowledgeable expert assistant. "
    "Answer questions thoroughly and confidently based on your domain expertise."
)

FOLLOWUP_PROMPTS = [
    "Are you confident in that answer? Please elaborate on your reasoning step by step.",
    "Is there anything about this topic you are uncertain about? Reflect carefully.",
]


def build_llm(model: str, temperature: float, top_logprobs: int) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        logprobs=True,
        top_logprobs=top_logprobs,
    )


def run_question(
    question_data: dict,
    model: str,
    turns: int = 1,
    temperature: float = 0.0,
    top_logprobs: int = 5,
) -> list:
    """
    Run a single question through the model for `turns` conversation turns.

    Returns a list of dicts, one per turn, each containing the response
    text, full logprobs sequence, and metadata.
    """
    llm = build_llm(model, temperature, top_logprobs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question_data["question"]),
    ]

    turn_results = []

    for turn_num in range(1, turns + 1):
        if turn_num > 1:
            # Append the previous AI response and a follow-up prompt
            prev_response = turn_results[-1]["response_text"]
            messages.append(AIMessage(content=prev_response))
            followup = FOLLOWUP_PROMPTS[min(turn_num - 2, len(FOLLOWUP_PROMPTS) - 1)]
            messages.append(HumanMessage(content=followup))

        response = llm.invoke(messages)
        metadata = response.response_metadata

        turn_results.append({
            "turn": turn_num,
            "prompt": messages[-1].content,
            "context_messages": [
                {"role": _msg_role(m), "content": m.content}
                for m in messages
            ],
            "response_text": response.content,
            "logprobs": _extract_logprobs(response),
            "finish_reason": metadata.get("finish_reason"),
            "prompt_tokens": metadata.get("token_usage", {}).get("prompt_tokens"),
            "completion_tokens": metadata.get("token_usage", {}).get("completion_tokens"),
            "total_tokens": metadata.get("token_usage", {}).get("total_tokens"),
            "model_name": metadata.get("model_name", model),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    return turn_results


def _extract_logprobs(response) -> list:
    """
    Extract logprob data from a LangChain ChatOpenAI response.

    Returns a list of per-token dicts:
      {
        "token": str,
        "logprob": float,          # log P(token | context)
        "top_logprobs": [          # top-k alternative tokens at this position
          {"token": str, "logprob": float}, ...
        ]
      }

    The "bytes" field is omitted intentionally to keep the dataset compact.
    """
    logprobs_meta = response.response_metadata.get("logprobs", {})
    content = logprobs_meta.get("content") or []

    result = []
    for token_info in content:
        result.append({
            "token": token_info.get("token"),
            "logprob": token_info.get("logprob"),
            "top_logprobs": [
                {"token": alt.get("token"), "logprob": alt.get("logprob")}
                for alt in token_info.get("top_logprobs", [])
            ],
        })
    return result


def _msg_role(msg: BaseMessage) -> str:
    """Map a LangChain message type to a role string."""
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, AIMessage):
        return "assistant"
    return "user"
