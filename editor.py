"""ИИ-редактор: чинит машинный перевод, сверяясь с оригиналом, и ужимает его под тайминг.

Работает батчами через OpenAI-совместимый эндпоинт YandexGPT. Жёсткое правило:
число и порядок строк не меняются — они привязаны к таймкодам. Если батч вернул
не то число строк или запрос упал, батч откатывается на машинный перевод.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

import aiohttp

import config

if TYPE_CHECKING:
    from speechkit import Phrase

log = logging.getLogger(__name__)

Progress = Callable[[str], Awaitable[None]]

SEP = "⟩"
LINE_RE = re.compile(rf"^\s*(\d+)\s*{SEP}\s*(.*)$")

SYSTEM_PROMPT = """Ты редактор дубляжа. На входе — расшифровка речи на исходном языке
и её машинный перевод на русский. Твоя работа — превратить машинный перевод в живую
устную русскую речь, которую человек произнесёт вслух.

Правила:
1. СВЕРЯЙСЯ С ОРИГИНАЛОМ. Распознавание речи ошибается: имена, названия и термины
   часто искажены. Восстанавливай их по смыслу оригинала, а не по кальке перевода.
2. Пиши так, как говорят вслух: короткие предложения, живой порядок слов,
   глаголы вместо отглагольных существительных.
3. Убирай канцелярит, штампы и воду. Но не пересказывай и не сокращай смысл —
   естественность важнее максимальной краткости.
4. Держи термины единообразными по глоссарию, если он дан.
5. Числа, единицы измерения и проценты пиши словами так, как их читают вслух.
6. Никаких пояснений, скобок от себя, кавычек-обёрток и markdown.

ФОРМАТ ОТВЕТА — САМОЕ ВАЖНОЕ:
Верни РОВНО столько строк, сколько было на входе, в том же порядке.
Каждая строка: номер, символ {sep}, отредактированный русский текст.
Ничего кроме этих строк не пиши.

У каждой строки указан бюджет символов — это длина, которая укладывается в её
тайм-слот. Старайся уложиться в бюджет: так фразу не придётся ускорять при озвучке.
Если уложиться нельзя без потери смысла — оставь смысл.""".replace("{sep}", SEP)


def load_glossary(path: str | Path | None = None) -> dict[str, str]:
    """Читает глоссарий EN→RU. Формат строки: `term = перевод` или `term|перевод`."""
    file = Path(path or config.GLOSSARY_FILE)
    if not file.exists():
        return {}
    terms: dict[str, str] = {}
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("|", "=", "\t"):
            if sep in line:
                left, right = line.split(sep, 1)
                left, right = left.strip(), right.strip()
                if left and right:
                    terms[left] = right
                break
    return terms


def _relevant_terms(glossary: dict[str, str], sources: list[str], limit: int = 40) -> list[str]:
    """Подмешиваем в промт только те термины, что реально встретились в батче."""
    blob = " ".join(sources).lower()
    hits = [f"{en} — {ru}" for en, ru in glossary.items() if en.lower() in blob]
    return hits[:limit]


def _budget(phrase: "Phrase") -> int:
    slot = phrase.slot
    if slot <= 0:
        return max(40, len(phrase.machine))
    return max(20, int(slot * config.CHARS_PER_SEC))


def _build_user_message(batch: list["Phrase"], glossary: dict[str, str]) -> str:
    parts: list[str] = []
    terms = _relevant_terms(glossary, [p.source for p in batch])
    if terms:
        parts.append("ГЛОССАРИЙ (держи эти соответствия):\n" + "\n".join(terms) + "\n")
    parts.append(f"СТРОК НА ВХОДЕ: {len(batch)}. Верни ровно столько же.\n")
    for i, phrase in enumerate(batch, 1):
        parts.append(
            f"{i}{SEP}бюджет {_budget(phrase)} симв.\n"
            f"  ОРИГИНАЛ: {phrase.source}\n"
            f"  ПЕРЕВОД: {phrase.machine}"
        )
    return "\n".join(parts)


def _parse_reply(text: str, expected: int) -> list[str] | None:
    lines: dict[int, str] = {}
    for raw in text.splitlines():
        match = LINE_RE.match(raw)
        if not match:
            continue
        idx = int(match.group(1))
        value = match.group(2).strip()
        if 1 <= idx <= expected and value:
            lines[idx] = value
    if len(lines) != expected:
        return None
    return [lines[i] for i in range(1, expected + 1)]


async def _ask(session: aiohttp.ClientSession, user_message: str) -> str:
    payload = {
        "model": f"gpt://{config.YC_FOLDER_ID}/{config.LLM_MODEL}",
        "temperature": config.LLM_TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }
    url = config.LLM_API_BASE.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Api-Key {config.LLM_API_KEY}"}
    async with session.post(url, json=payload, headers=headers) as resp:
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"{resp.status}: {body[:300]}")
        import json

        data = json.loads(body)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Модель вернула пустой ответ: {body[:200]}")
    return choices[0].get("message", {}).get("content", "") or ""


async def edit_phrases(
    phrases: list["Phrase"], progress: Progress | None = None
) -> tuple[bool, int]:
    """Правит phrases[*].text на месте. Возвращает (редактор отработал, сколько батчей упало)."""
    if not config.LLM_EDIT or not phrases:
        return False, 0

    glossary = load_glossary()
    batches = [
        phrases[i : i + config.LLM_BATCH] for i in range(0, len(phrases), config.LLM_BATCH)
    ]
    failed = 0
    timeout = aiohttp.ClientTimeout(total=config.LLM_TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for n, batch in enumerate(batches, 1):
            message = _build_user_message(batch, glossary)
            edited: list[str] | None = None
            for attempt in range(2):
                try:
                    reply = await _ask(session, message)
                    edited = _parse_reply(reply, len(batch))
                    if edited:
                        break
                    log.warning("Батч %s: модель вернула не то число строк", n)
                except Exception as exc:  # noqa: BLE001 — падение батча не рушит синхрон
                    log.warning("Батч %s не прошёл: %s", n, exc)
                await asyncio.sleep(2 * (attempt + 1))

            if edited:
                for phrase, text in zip(batch, edited):
                    phrase.text = text
            else:
                failed += 1  # откат на машинный перевод: phrase.text уже равен machine

            if progress and n % 5 == 0:
                await progress(f"Вычитываю перевод… {n} из {len(batches)}")

    return True, failed
