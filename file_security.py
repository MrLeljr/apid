"""Attachment preflight checks for multimodal LLM requests."""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from typing import Any, Iterable


DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[-\w.+/]+)?(?:;[-\w=.+]+)*;base64,(?P<data>[A-Za-z0-9+/=\s]+)$")
BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/=\s]{80,}$")
PRINTABLE_PATTERN = re.compile(rb"[ -~]{8,}")
FILE_DATA_KEYS = {
    "base64",
    "b64",
    "data",
    "file_data",
    "file_bytes",
    "bytes",
    "content_base64",
}
FILE_HINT_KEYS = {
    "filename",
    "file_name",
    "name",
    "mime_type",
    "media_type",
    "type",
}
ACTIVE_SIGNATURES = (
    (b"/JavaScript", "pdf_javascript"),
    (b"/JS", "pdf_javascript"),
    (b"/OpenAction", "pdf_open_action"),
    (b"/AA", "pdf_additional_action"),
    (b"/Launch", "pdf_launch_action"),
    (b"/EmbeddedFile", "pdf_embedded_file"),
    (b"/RichMedia", "pdf_rich_media"),
    (b"vbaProject.bin", "office_macro"),
    (b"word/vbaData.xml", "office_macro"),
    (b"<script", "embedded_script"),
    (b"powershell", "embedded_shell_command"),
    (b"cmd.exe", "embedded_shell_command"),
)
EXECUTABLE_MAGIC = (
    (b"MZ", "windows_executable"),
    (b"\x7fELF", "elf_executable"),
    (b"#!/", "script_file"),
)


@dataclass(frozen=True)
class FileCandidate:
    """Decoded file-like payload found in a request body."""

    path: str
    data: bytes
    mime_type: str | None = None
    filename: str | None = None


def _looks_like_base64(value: str) -> bool:
    compact = "".join(value.split())
    return len(compact) >= 80 and len(compact) % 4 == 0 and bool(BASE64_PATTERN.match(value))


def _decode_base64(value: str) -> bytes | None:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        try:
            return base64.b64decode("".join(value.split()), validate=True)
        except (binascii.Error, ValueError):
            return None


def _value_as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _metadata_from_node(node: dict[str, Any]) -> tuple[str | None, str | None]:
    mime_type = _value_as_text(node.get("mime_type")) or _value_as_text(node.get("media_type"))
    filename = _value_as_text(node.get("filename")) or _value_as_text(node.get("file_name")) or _value_as_text(node.get("name"))
    return mime_type or None, filename or None


def _candidate_from_string(value: str, path: str, *, mime_type: str | None = None, filename: str | None = None) -> FileCandidate | None:
    data_url_match = DATA_URL_PATTERN.match(value.strip())
    if data_url_match:
        decoded = _decode_base64(data_url_match.group("data"))
        if decoded is None:
            return None
        return FileCandidate(path=path, data=decoded, mime_type=mime_type or data_url_match.group("mime"), filename=filename)

    if _looks_like_base64(value):
        decoded = _decode_base64(value)
        if decoded is None:
            return None
        return FileCandidate(path=path, data=decoded, mime_type=mime_type, filename=filename)

    return None


def iter_file_candidates(payload: Any, path: str = "$") -> Iterable[FileCandidate]:
    """Yield decoded file-like values from OpenAI-style request payloads."""

    if isinstance(payload, dict):
        mime_type, filename = _metadata_from_node(payload)

        image_url = payload.get("image_url")
        if isinstance(image_url, str):
            candidate = _candidate_from_string(image_url, f"{path}.image_url", mime_type=mime_type, filename=filename)
            if candidate:
                yield candidate
        elif isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str):
                candidate = _candidate_from_string(url, f"{path}.image_url.url", mime_type=mime_type, filename=filename)
                if candidate:
                    yield candidate

        for key, value in payload.items():
            child_path = f"{path}.{key}"
            if key in FILE_DATA_KEYS and isinstance(value, str):
                candidate = _candidate_from_string(value, child_path, mime_type=mime_type, filename=filename)
                if candidate:
                    yield candidate
            elif isinstance(value, (dict, list)):
                yield from iter_file_candidates(value, child_path)
            elif key in FILE_HINT_KEYS:
                continue

    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            yield from iter_file_candidates(item, f"{path}[{index}]")


def detect_magic(data: bytes) -> str:
    """Return a coarse file type based on magic bytes."""

    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"PK\x03\x04"):
        return "application/zip"
    for magic, label in EXECUTABLE_MAGIC:
        if data.startswith(magic):
            return label
    return "application/octet-stream"


def extract_printable_text(data: bytes, max_chars: int) -> str:
    """Extract readable strings from binary content for prompt-injection scanning."""

    fragments: list[str] = []
    remaining = max_chars
    for match in PRINTABLE_PATTERN.finditer(data):
        if remaining <= 0:
            break
        text = match.group(0).decode("latin-1", errors="ignore")
        fragments.append(text[:remaining])
        remaining -= len(fragments[-1])
    return "\n".join(fragments)


def inspect_file_candidate(candidate: FileCandidate, *, max_bytes: int, max_extracted_chars: int) -> dict[str, Any]:
    """Inspect one decoded attachment and return a structured result."""

    magic_type = detect_magic(candidate.data)
    findings: list[dict[str, str]] = []
    declared_type = (candidate.mime_type or "").lower()
    filename = (candidate.filename or "").lower()

    if len(candidate.data) > max_bytes:
        findings.append({"type": "oversized_attachment", "detail": f"decoded file is {len(candidate.data)} bytes"})

    for magic, label in EXECUTABLE_MAGIC:
        if candidate.data.startswith(magic):
            findings.append({"type": "executable_content", "detail": label})

    if declared_type.startswith(("image/", "application/pdf")) and magic_type in {
        "windows_executable",
        "elf_executable",
        "script_file",
    }:
        findings.append({"type": "file_type_mismatch", "detail": f"declared {declared_type}, detected {magic_type}"})

    if filename.endswith((".jpg", ".jpeg", ".png", ".gif", ".pdf")) and magic_type in {
        "windows_executable",
        "elf_executable",
        "script_file",
    }:
        findings.append({"type": "file_type_mismatch", "detail": f"filename {filename}, detected {magic_type}"})

    lowered_data = candidate.data.lower()
    for signature, label in ACTIVE_SIGNATURES:
        if signature.lower() in lowered_data:
            findings.append({"type": "active_embedded_content", "detail": label})

    return {
        "path": candidate.path,
        "filename": candidate.filename,
        "declared_mime_type": candidate.mime_type,
        "detected_type": magic_type,
        "size_bytes": len(candidate.data),
        "findings": findings,
        "extracted_text": extract_printable_text(candidate.data, max_extracted_chars),
    }
