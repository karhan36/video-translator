"""Текстовые форматы ответа: субтитры .srt и чистый текст .txt с абзацами."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from speechkit import Phrase


def _timecode(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(phrases: list["Phrase"], path: Path) -> Path:
    blocks: list[str] = []
    for i, phrase in enumerate(phrases, 1):
        text = (phrase.text or phrase.machine).strip()
        if not text:
            continue
        end = phrase.end if phrase.end > phrase.start else phrase.start + 2.0
        blocks.append(
            f"{i}\n{_timecode(phrase.start)} --> {_timecode(end)}\n{text}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def build_paragraphs(phrases: list["Phrase"]) -> list[str]:
    """Новый абзац — при паузе дольше PARA_GAP или когда абзац перерос PARA_MAX_CHARS."""
    paragraphs: list[str] = []
    current: list[str] = []
    length = 0
    prev_end: float | None = None

    for phrase in phrases:
        text = (phrase.text or phrase.machine).strip()
        if not text:
            continue

        gap = phrase.start - prev_end if prev_end is not None else 0.0
        if current and (gap >= config.PARA_GAP or length >= config.PARA_MAX_CHARS):
            paragraphs.append(" ".join(current))
            current, length = [], 0

        current.append(text)
        length += len(text) + 1
        prev_end = phrase.end

    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def write_txt(phrases: list["Phrase"], path: Path, title: str = "") -> Path:
    parts = []
    if title:
        parts.append(title.strip())
        parts.append("")
    parts.extend(build_paragraphs(phrases))
    path.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")
    return path
