#!/usr/bin/env python3
"""AANA custom scaffold for tau2-bench text evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from tau2.data_model.simulation import TextRunConfig
from tau2.environment.tool import Tool
from tau2.registry import registry
from tau2.runner.batch import run_domain
from tau2.utils.llm_utils import generate


AANA_REPO = Path(r"C:\Users\soria\OneDrive\Documents\New project").resolve()
if str(AANA_REPO) not in sys.path:
    sys.path.insert(0, str(AANA_REPO))

from eval_pipeline.pre_tool_call_gate import gate_pre_tool_call  # noqa: E402


SYSTEM_PROMPT = """\
You are a customer service agent that helps the user according to the policy.
You may either respond to the user or call one provided tool, but not both in
the same turn.

This is a custom AANA scaffold. Before any proposed tool call is executed, the
runtime checks the proposed call using a structured pre-tool-call contract:
tool category, authorization state, evidence references, risk domain, and route.

If policy evidence, user intent, authorization, or confirmation is missing, ask
the user instead of guessing. If the requested action is policy-supported and
the user has provided the needed information or confirmation, proceed.

<policy>
{domain_policy}
</policy>
"""


class AANAContractAgentState:
    def __init__(
        self,
        system_messages: list[SystemMessage],
        messages: list[APICompatibleMessage],
        gate_records: Optional[list[dict]] = None,
    ):
        self.system_messages = system_messages
        self.messages = messages
        self.gate_records = gate_records or []


def _latest_user_summary(messages: list[APICompatibleMessage]) -> str:
    for message in reversed(messages):
        if getattr(message, "role", None) == "user" and getattr(message, "content", None):
            return str(message.content)[:700]
    return "No user message available."


def _tool_category(tool_name: str) -> str:
    name = tool_name.lower()
    if any(token in name for token in (
        "book", "cancel", "change", "create", "delete", "modify", "purchase",
        "refund", "reset", "send", "transfer", "update", "verify", "apply",
        "submit", "pay", "escalate",
    )):
        return "write"
    if any(token in name for token in ("get", "lookup", "retrieve", "search", "find", "list")):
        if any(token in name for token in ("customer", "user", "account", "order", "reservation", "profile")):
            return "private_read"
        return "public_read"
    return "unknown"


def _risk_domain(domain_policy: str) -> str:
    text = domain_policy.lower()
    if any(token in text for token in ("bank", "payment", "account", "card", "wire transfer")):
        return "finance"
    if any(token in text for token in ("flight", "airline", "reservation")):
        return "customer_support"
    if any(token in text for token in ("telecom", "phone", "wireless", "internet")):
        return "customer_support"
    return "commerce"


def _auth_state(tool_category: str, messages: list[APICompatibleMessage]) -> str:
    if tool_category == "public_read":
        return "confirmed"
    user_text = _latest_user_summary(messages).lower()
    if any(token in user_text for token in (
        "yes", "confirm", "confirmed", "go ahead", "please", "do it", "i want",
        "book", "cancel", "change", "refund", "update", "send", "transfer",
        "my", "i need", "i would like",
    )):
        return "confirmed"
    return "authenticated"


def _gate_event(
    *,
    tool_name: str,
    arguments: dict,
    domain_policy: str,
    messages: list[APICompatibleMessage],
) -> dict:
    category = _tool_category(tool_name)
    return {
        "schema_version": "aana.agent_tool_precheck.v1",
        "tool_name": tool_name,
        "tool_category": category,
        "authorization_state": _auth_state(category, messages),
        "evidence_refs": [
            {
                "source_id": "tau2.domain_policy",
                "kind": "policy",
                "trust_tier": "verified",
                "redaction_status": "redacted",
                "summary": domain_policy[:700],
            },
            {
                "source_id": "tau2.latest_user_message",
                "kind": "user_message",
                "trust_tier": "runtime",
                "redaction_status": "redacted",
                "summary": _latest_user_summary(messages),
            },
        ],
        "risk_domain": _risk_domain(domain_policy),
        "proposed_arguments": arguments,
        "recommended_route": "accept",
        "user_intent": _latest_user_summary(messages),
    }


def _blocked_response(tool_name: str, decision: dict) -> AssistantMessage:
    route = decision.get("recommended_action", "defer")
    if route == "ask":
        content = "I need one more confirmation before I can perform that action. Please confirm what you want me to do."
    elif route == "refuse":
        content = "I cannot perform that action because the required policy, evidence, or authorization check did not pass."
    else:
        content = "I need to verify the required policy, evidence, or authorization before performing that action."
    return AssistantMessage.text(
        content=content,
        raw_data={"aana_gate": {"tool_name": tool_name, **decision}},
    )


class AANAContractAgent(HalfDuplexAgent[AANAContractAgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "openai/gpt-4.1-mini",
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = llm_args or {}
        self._tool_names = {tool.name for tool in tools}

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> AANAContractAgentState:
        return AANAContractAgentState(
            system_messages=[SystemMessage(role="system", content=SYSTEM_PROMPT.format(domain_policy=self.domain_policy))],
            messages=list(message_history) if message_history else [],
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: AANAContractAgentState,
    ) -> tuple[AssistantMessage, AANAContractAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            call_name="aana_contract_agent_response",
            **self.llm_args,
        )

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.name not in self._tool_names:
                    response = _blocked_response(tool_call.name, {
                        "recommended_action": "defer",
                        "hard_blockers": ["unknown_tool"],
                        "reasons": ["tool_not_registered"],
                    })
                    break
                decision = gate_pre_tool_call(
                    _gate_event(
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        domain_policy=self.domain_policy,
                        messages=state.messages,
                    )
                )
                state.gate_records.append(decision)
                if decision["recommended_action"] != "accept":
                    response = _blocked_response(tool_call.name, decision)
                    break

        response.raw_data = response.raw_data or {}
        response.raw_data["aana_gate_records"] = state.gate_records[-5:]
        state.messages.append(response)
        return response, state


def create_aana_contract_agent(tools, domain_policy, **kwargs):
    return AANAContractAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm", "openai/gpt-4.1-mini"),
        llm_args=kwargs.get("llm_args"),
    )


def register_aana_agent() -> None:
    if "aana_contract_agent" not in registry.get_agents():
        registry.register_agent_factory(create_aana_contract_agent, "aana_contract_agent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--agent-llm", default="openai/gpt-4.1-mini")
    parser.add_argument("--user-llm", default="openai/gpt-4.1-mini")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--task-ids", nargs="*")
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retrieval-config")
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()

    register_aana_agent()
    config = TextRunConfig(
        domain=args.domain,
        agent="aana_contract_agent",
        llm_agent=args.agent_llm,
        llm_user=args.user_llm,
        num_trials=args.num_trials,
        num_tasks=args.num_tasks,
        task_ids=args.task_ids,
        save_to=args.save_to,
        max_concurrency=args.max_concurrency,
        timeout=args.timeout,
        retrieval_config=args.retrieval_config,
        auto_resume=args.auto_resume,
    )
    results = run_domain(config)
    print(json.dumps({
        "save_to": args.save_to,
        "domain": args.domain,
        "num_simulations": len(results.simulations),
    }, indent=2))


if __name__ == "__main__":
    main()
