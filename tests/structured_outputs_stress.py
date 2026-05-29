#!/usr/bin/env python3
"""Stress JSON structured outputs on OpenAI-compatible chat/responses APIs.

Examples:
  python3 tests/structured_outputs_stress.py \
      --base-url http://127.0.0.1:8000/v1 --model ds4 --apis chat,responses

  python3 tests/structured_outputs_stress.py \
      --base-url http://127.0.0.1:8080/v1 --model qwen --apis chat
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    schema: dict[str, Any] | None
    json_object: bool = False


CASES: list[Case] = [
    Case(
        name="calendar_event",
        prompt=(
            "Create one calendar event for Alice and Bob having lunch on "
            "2026-06-01 at noon. Return only the requested JSON object."
        ),
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "date": {"type": "string"},
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                },
            },
            "required": ["name", "date", "participants"],
            "additionalProperties": False,
        },
    ),
    Case(
        name="enum_const_integer_boolean",
        prompt=(
            "Return a compact health-check result. Use status ok, one priority, "
            "a retry count, and whether the system is active."
        ),
        schema={
            "type": "object",
            "properties": {
                "status": {"const": "ok"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "retry_count": {"type": "integer", "minimum": 0, "maximum": 5},
                "active": {"type": "boolean"},
            },
            "required": ["status", "priority", "retry_count", "active"],
            "additionalProperties": False,
        },
    ),
    Case(
        name="nested_arrays",
        prompt=(
            "Return a 2 by 2 integer matrix and two short labels. Keep values "
            "small and return only JSON."
        ),
        schema={
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "integer", "minimum": -9, "maximum": 9},
                    },
                },
                "labels": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "string"},
                },
            },
            "required": ["matrix", "labels"],
            "additionalProperties": False,
        },
    ),
    Case(
        name="nullable_anyof_number_bounds",
        prompt=(
            "Return a score between zero and one, and use either an owner name "
            "or null if unknown."
        ),
        schema={
            "type": "object",
            "properties": {
                "owner": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["owner", "score"],
            "additionalProperties": False,
        },
    ),
    Case(
        name="pattern_string",
        prompt="Return an inventory code in the form two uppercase letters, dash, three digits.",
        schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "pattern": "^[A-Z]{2}-[0-9]{3}$"}
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    ),
    Case(
        name="json_object_mode",
        prompt="Return a JSON object with two fields describing a tiny task list.",
        schema=None,
        json_object=True,
    ),
]


class ValidationError(Exception):
    pass


def type_matches(value: Any, typ: str) -> bool:
    if typ == "object":
        return isinstance(value, dict)
    if typ == "array":
        return isinstance(value, list)
    if typ == "string":
        return isinstance(value, str)
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if typ == "boolean":
        return isinstance(value, bool)
    if typ == "null":
        return value is None
    return True


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    if "anyOf" in schema:
        errors: list[str] = []
        for option in schema["anyOf"]:
            try:
                validate_schema(value, option, path)
                return
            except ValidationError as exc:
                errors.append(str(exc))
        raise ValidationError(f"{path}: did not match anyOf: {'; '.join(errors)}")

    if "const" in schema and value != schema["const"]:
        raise ValidationError(f"{path}: expected const {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path}: expected one of {schema['enum']!r}, got {value!r}")

    typ = schema.get("type")
    if isinstance(typ, list):
        if not any(type_matches(value, t) for t in typ):
            raise ValidationError(f"{path}: wrong type {type(value).__name__}, expected {typ}")
    elif isinstance(typ, str) and not type_matches(value, typ):
        raise ValidationError(f"{path}: wrong type {type(value).__name__}, expected {typ}")

    if typ == "object" or "properties" in schema:
        if not isinstance(value, dict):
            raise ValidationError(f"{path}: expected object")
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ValidationError(f"{path}: missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                raise ValidationError(f"{path}: extra properties {extra!r}")
        for key, sub in props.items():
            if key in value:
                validate_schema(value[key], sub, f"{path}.{key}")

    if typ == "array" or "items" in schema:
        if not isinstance(value, list):
            raise ValidationError(f"{path}: expected array")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            raise ValidationError(f"{path}: expected at least {min_items} items")
        if max_items is not None and len(value) > max_items:
            raise ValidationError(f"{path}: expected at most {max_items} items")
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                validate_schema(item, items, f"{path}[{i}]")

    if isinstance(value, str) and "pattern" in schema:
        if re.fullmatch(schema["pattern"], value) is None:
            raise ValidationError(f"{path}: {value!r} does not match {schema['pattern']!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError(f"{path}: {value!r} is below minimum {schema['minimum']!r}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValidationError(f"{path}: {value!r} is above maximum {schema['maximum']!r}")


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON response: {raw[:1000]}") from exc
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(f"API error: {body['error']!r}")
    return body


def chat_payload(model: str, case: Case, json_object_schema: bool) -> dict[str, Any]:
    response_format: dict[str, Any]
    if case.json_object:
        response_format = {"type": "json_object"}
        if json_object_schema:
            response_format["schema"] = {"type": "object"}
    else:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": case.name,
                "strict": True,
                "schema": case.schema,
            },
        }
    return {
        "model": model,
        "messages": [{"role": "user", "content": case.prompt}],
        "max_tokens": 256,
        "temperature": 0,
        "response_format": response_format,
    }


def responses_payload(model: str, case: Case, json_object_schema: bool) -> dict[str, Any]:
    fmt: dict[str, Any]
    if case.json_object:
        fmt = {"type": "json_object"}
        if json_object_schema:
            fmt["schema"] = {"type": "object"}
    else:
        fmt = {
            "type": "json_schema",
            "name": case.name,
            "strict": True,
            "schema": case.schema,
        }
    return {
        "model": model,
        "input": case.prompt,
        "max_output_tokens": 256,
        "temperature": 0,
        "text": {"format": fmt},
    }


def extract_chat_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"missing choices in chat response: {body!r}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
        return "".join(out)
    raise RuntimeError(f"missing text content in chat response: {body!r}")


def extract_responses_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    out: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content", []):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    out.append(part["text"])
    if out:
        return "".join(out)
    raise RuntimeError(f"missing output text in responses response: {body!r}")


def check_case(
    api: str,
    base_url: str,
    model: str,
    case: Case,
    timeout: float,
    json_object_schema: bool,
) -> str:
    if api == "chat":
        body = post_json(
            f"{base_url}/chat/completions",
            chat_payload(model, case, json_object_schema),
            timeout,
        )
        text = extract_chat_text(body)
    elif api == "responses":
        body = post_json(
            f"{base_url}/responses",
            responses_payload(model, case, json_object_schema),
            timeout,
        )
        text = extract_responses_text(body)
    else:
        raise RuntimeError(f"unknown api {api!r}")

    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{api}/{case.name}: output is not JSON: {text!r}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{api}/{case.name}: output is not a JSON object: {value!r}")
    if case.schema is not None:
        validate_schema(value, case.schema)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True, help="Base URL, usually http://host:port/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--apis", default="chat,responses", help="Comma-separated: chat,responses")
    p.add_argument("--case", action="append", help="Run only this case name; may repeat")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument(
        "--json-object-schema",
        action="store_true",
        help="Send {'type':'object'} with json_object mode for servers that require a concrete schema.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    apis = [x.strip() for x in args.apis.split(",") if x.strip()]
    selected = set(args.case or [])
    cases = [c for c in CASES if not selected or c.name in selected]
    missing = selected - {c.name for c in CASES}
    if missing:
        print(f"unknown case(s): {', '.join(sorted(missing))}", file=sys.stderr)
        return 2

    failures = 0
    for repeat in range(args.repeat):
        for api in apis:
            for case in cases:
                label = f"{api}/{case.name}"
                if args.repeat > 1:
                    label = f"{label}#{repeat + 1}"
                t0 = time.time()
                try:
                    value = check_case(
                        api,
                        base_url,
                        args.model,
                        case,
                        args.timeout,
                        args.json_object_schema,
                    )
                    elapsed = time.time() - t0
                    if args.verbose:
                        print(f"PASS {label} {elapsed:.2f}s {value}")
                    else:
                        print(f"PASS {label} {elapsed:.2f}s")
                except Exception as exc:
                    failures += 1
                    print(f"FAIL {label}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
