#!/usr/bin/env python3
"""
End-to-end benchmark example:
1) Ensure scenarios exist
2) Ensure requested agents exist
3) Run benchmark
4) Print by-model aggregate scores
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class AgentSpec:
    provider: str
    model_name: str


def parse_agent_spec(value: str) -> AgentSpec:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid --agent-spec '{value}'. Expected format provider:model_name"
        )
    provider, model_name = value.split(":", 1)
    provider = provider.strip().lower()
    model_name = model_name.strip()
    if not provider or not model_name:
        raise argparse.ArgumentTypeError(
            f"Invalid --agent-spec '{value}'. provider and model_name are required."
        )
    return AgentSpec(provider=provider, model_name=model_name)


def ensure_scenarios(client: httpx.Client) -> None:
    response = client.post("/api/v1/scenarios/generate")
    response.raise_for_status()
    payload = response.json()
    print(
        f"[scenarios] generated={payload.get('total_generated')} inserted={payload.get('inserted')}",
        flush=True,
    )


def get_or_create_agent(client: httpx.Client, spec: AgentSpec, temperature: float, max_tokens: int) -> int:
    response = client.get("/api/v1/agents")
    response.raise_for_status()
    agents = response.json()

    for agent in agents:
        if agent.get("provider") == spec.provider and agent.get("model_name") == spec.model_name:
            return int(agent["id"])

    create_response = client.post(
        "/api/v1/agents",
        json={
            "model_name": spec.model_name,
            "provider": spec.provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    create_response.raise_for_status()
    created = create_response.json()
    return int(created["id"])


def select_scenarios(client: httpx.Client, scenario_count: int) -> list[int]:
    response = client.get("/api/v1/scenarios")
    response.raise_for_status()
    scenarios = response.json()
    selected = [int(row["id"]) for row in scenarios[:scenario_count]]
    if not selected:
        raise RuntimeError("No scenarios available after generation.")
    return selected


def print_results(rows: list[dict]) -> None:
    if not rows:
        print("[results] no rows returned")
        return

    header = f"{'agent_id':>8}  {'model':<34}  {'bias_type':<16}  {'mean':>6}  {'n':>5}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['agent_id']):>8}  "
            f"{str(row['model_name'])[:34]:<34}  "
            f"{str(row['bias_type'])[:16]:<16}  "
            f"{float(row['mean_bias_score']):>6.3f}  "
            f"{int(row['sample_count']):>5}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an end-to-end benchmark flow against the API.")
    parser.add_argument("--host", default="http://localhost:8000", help="API host (default: http://localhost:8000)")
    parser.add_argument("--scenario-count", type=int, default=8, help="Number of scenarios to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Agent temperature for new agents")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Agent max tokens for new agents")
    parser.add_argument(
        "--agent-spec",
        type=parse_agent_spec,
        action="append",
        default=None,
        help="Agent to include (repeatable), format provider:model_name",
    )
    args = parser.parse_args()

    agent_specs = args.agent_spec or [AgentSpec("nvidia", "meta/llama-3.1-70b-instruct")]

    with httpx.Client(base_url=args.host, timeout=120) as client:
        ensure_scenarios(client)
        scenario_ids = select_scenarios(client, scenario_count=args.scenario_count)
        print(f"[scenarios] selected={scenario_ids}", flush=True)

        agent_ids = []
        for spec in agent_specs:
            agent_id = get_or_create_agent(
                client,
                spec=spec,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            agent_ids.append(agent_id)
            print(f"[agents] {spec.provider}:{spec.model_name} -> id={agent_id}", flush=True)

        run_response = client.post(
            "/api/v1/benchmark/run",
            json={"agent_ids": agent_ids, "scenario_ids": scenario_ids},
        )
        run_response.raise_for_status()
        run_payload = run_response.json()
        run_id = run_payload["run_id"]
        print(f"[benchmark] run_id={run_id} evaluations={run_payload['evaluations_run']}", flush=True)

        results_response = client.get("/api/v1/results/by-model", params={"run_id": run_id})
        results_response.raise_for_status()
        rows = results_response.json()
        print_results(rows)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
