# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, Sequence
from urllib.parse import urlparse, urlunparse

EndpointKind = Literal["chat", "completions"]


@dataclass(frozen=True)
class Endpoint:
    original: str
    base_url: str
    chat_url: str
    completions_url: str
    models_url: str
    kind: EndpointKind


@dataclass(frozen=True)
class Completion:
    text: str
    n_tokens: int | None
    finish_reason: str | None
    model: str | None
    raw: dict[str, Any]


class EndpointError(RuntimeError):
    pass


_PAYLOAD_LOCK = frozenset({"model", "messages", "prompt", "temperature", "top_p", "max_tokens", "stream"})
_CONNECTION_MARKERS = (
    "connection refused",
    "nodename nor servname",
    "name or service not known",
    "timed out",
    "timeout",
    "network is unreachable",
    "connection reset",
    "failed to resolve",
    "errno 61",
    "errno 111",
)


def resolve_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for name in ("NVIDIA_API_KEY", "NGC_API_KEY", "OPENAI_API_KEY", "API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def normalize_model_url(model_url: str) -> Endpoint:
    raw = (model_url or "").strip()
    if not raw:
        raise ValueError("model-url is empty")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("model-url must be an HTTP(S) OpenAI-compatible endpoint")

    path = parsed.path.rstrip("/")
    lower = path.lower()
    kind: EndpointKind = "chat"
    if lower.endswith("/chat/completions"):
        root_path = path[: -len("/chat/completions")]
    elif lower.endswith("/completions"):
        root_path = path[: -len("/completions")]
        kind = "completions"
    else:
        root_path = path
        if not root_path.endswith("/v1"):
            root_path = f"{root_path}/v1" if root_path else "/v1"

    def _join(suffix: str) -> str:
        return urlunparse(
            (parsed.scheme, parsed.netloc, f"{root_path}{suffix}", "", "", "")
        )

    base = urlunparse((parsed.scheme, parsed.netloc, root_path, "", "", ""))
    return Endpoint(
        original=raw,
        base_url=base,
        chat_url=_join("/chat/completions"),
        completions_url=_join("/completions"),
        models_url=_join("/models"),
        kind=kind,
    )


def build_payload(
    *,
    model: str,
    messages: Sequence[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    kind: EndpointKind = "chat",
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if kind == "chat":
        payload["messages"] = list(messages)
    else:
        payload["prompt"] = messages_to_prompt(messages)
    if extra_body:
        for key, value in extra_body.items():
            if key in _PAYLOAD_LOCK:
                continue
            payload[key] = value
    return payload


def is_multi_model_host(endpoint: Endpoint) -> bool:
    host = urlparse(endpoint.base_url).netloc.lower()
    return "nvidia.com" in host or "openai.com" in host


def retry_delay_s(code: int, attempt: int, headers: Any | None = None) -> float:
    """Backoff for retryable HTTP errors. 429 waits longer and honors Retry-After."""
    if int(code) == 429:
        delay = 20.0 * (attempt + 1)
        raw = None
        if headers is not None:
            raw = headers.get("Retry-After")
        if raw:
            try:
                delay = max(delay, float(raw))
            except (TypeError, ValueError):
                pass
        return min(delay, 120.0)
    return float(min(2 ** attempt, 8))


def is_connection_error(exc: BaseException) -> bool:
    blob = str(exc).lower()
    return any(marker in blob for marker in _CONNECTION_MARKERS)


def messages_to_prompt(messages: Sequence[dict[str, str]]) -> str:
    if len(messages) > 1 and messages[-1].get("role") == "assistant":
        head = "\n\n".join(msg.get("content", "") for msg in messages[:-1])
        return head + str(messages[-1].get("content") or "")
    return "\n\n".join(msg.get("content", "") for msg in messages)


def parse_completion(body: dict[str, Any]) -> Completion:
    choices = body.get("choices") or []
    if not choices:
        raise EndpointError(f"endpoint returned no choices: {body!r}"[:500])
    choice = choices[0]
    text = ""
    message = choice.get("message") or {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning and "</think>" not in text:
            if not text and choice.get("finish_reason") == "length":
                text = f"<think>{reasoning}"
            else:
                text = f"<think>{reasoning}</think>\n{text}"
    if not text:
        text = str(choice.get("text") or "")
    usage = body.get("usage") or {}
    n_tokens = usage.get("completion_tokens")
    if n_tokens is not None:
        try:
            n_tokens = int(n_tokens)
        except (TypeError, ValueError):
            n_tokens = None
    return Completion(
        text=text,
        n_tokens=n_tokens,
        finish_reason=choice.get("finish_reason"),
        model=body.get("model"),
        raw=body,
    )


class OpenAIChatClient:
    def __init__(
        self,
        endpoint: Endpoint,
        model_name: str,
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
        extra_body: dict[str, Any] | None = None,
        retries: int = 6,
    ) -> None:
        self.endpoint = endpoint
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})
        self.retries = retries

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Avoid Cloudflare rejecting urllib's default Python-urllib signature.
            "User-Agent": "curl/8.7.1",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_json(self, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        method = "GET" if payload is None else "POST"
        last_error: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                if not raw.strip():
                    return {}
                body = json.loads(raw)
                if isinstance(body, dict) and body.get("error"):
                    raise EndpointError(str(body["error"]))
                return body if isinstance(body, dict) else {"data": body}
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = EndpointError(f"HTTP {exc.code} from {url}: {detail[:800]}")
                if exc.code in {408, 409, 429, 500, 502, 503, 504} and attempt + 1 < self.retries:
                    time.sleep(retry_delay_s(exc.code, attempt, exc.headers))
                    continue
                raise last_error from exc
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                ConnectionError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                last_error = EndpointError(f"request to {url} failed: {exc}")
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** (attempt + 1), 30))
                    continue
                raise last_error from exc
        raise last_error or EndpointError(f"request to {url} failed")

    def list_models(self) -> list[str]:
        body = self._request_json(self.endpoint.models_url)
        rows = body.get("data") or body.get("models") or []
        ids: list[str] = []
        for row in rows:
            if isinstance(row, dict) and row.get("id"):
                ids.append(str(row["id"]))
            elif isinstance(row, str):
                ids.append(row)
        return ids

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        extra: dict[str, Any] | None = None,
    ) -> Completion:
        url = self.endpoint.chat_url if self.endpoint.kind == "chat" else self.endpoint.completions_url
        extra_body = dict(self.extra_body)
        if extra:
            extra_body.update(extra)
        payload = build_payload(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            kind=self.endpoint.kind,
            extra_body=extra_body,
        )
        body = self._request_json(url, payload)
        return parse_completion(body)


def detect_model_name(endpoint: Endpoint, api_key: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if is_multi_model_host(endpoint):
        raise EndpointError("this host serves many models; pass --model-name explicitly")
    client = OpenAIChatClient(endpoint, model_name="unknown", api_key=api_key, timeout=30.0, retries=1)
    try:
        ids = client.list_models()
    except EndpointError as exc:
        raise EndpointError(
            "could not auto-detect --model-name from /v1/models; pass --model-name explicitly"
        ) from exc
    if not ids:
        raise EndpointError("/v1/models returned no ids; pass --model-name")
    return ids[0]
