"""Helpers for loading Azure OpenAI settings from flags or environment."""

from __future__ import annotations

import os


AZURE_OPENAI_API_KEY_ENV = "AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT_ENV = "AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_DEPLOYMENT_ENV = "AZURE_OPENAI_DEPLOYMENT"
DEFAULT_AZURE_OPENAI_API_VERSION = "2025-03-01-preview"


def resolve_azure_openai_config(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    deployment: str | None = None,
) -> dict[str, str]:
    """Resolve required Azure OpenAI settings from explicit args or env vars."""

    resolved_api_key = api_key or os.environ.get(AZURE_OPENAI_API_KEY_ENV)
    resolved_endpoint = endpoint or os.environ.get(AZURE_OPENAI_ENDPOINT_ENV)
    resolved_deployment = deployment or os.environ.get(AZURE_OPENAI_DEPLOYMENT_ENV)

    missing = []
    if not resolved_api_key:
        missing.append(f"--api-key or {AZURE_OPENAI_API_KEY_ENV}")
    if not resolved_endpoint:
        missing.append(f"--endpoint or {AZURE_OPENAI_ENDPOINT_ENV}")
    if not resolved_deployment:
        missing.append(f"--deployment or {AZURE_OPENAI_DEPLOYMENT_ENV}")

    if missing:
        raise ValueError("Missing Azure OpenAI configuration: " + ", ".join(missing))

    return {
        "api_key": resolved_api_key,
        "endpoint": resolved_endpoint.rstrip("/") + "/",
        "deployment": resolved_deployment,
    }
