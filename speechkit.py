"""Движок перевода на Yandex SpeechKit: распознавание, перевод, синтез, укладка.

Пайплайн:
    yt-dlp → ffmpeg (OGG_OPUS 16 кГц моно) → Object Storage → STT longRunning
    → Translate → ИИ-редактор → TTS → укладка по таймкодам оригинала.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterable

import aiohttp
import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

import config
import editor
import media

log = logging.getLogger(__name__)

Progress = Callable[[str], Awaitable[None]]

BYTES_PER_SAMPLE = 2
TRANSLATE_MAX_CHARS = 9000  # с запасом к лимиту API в 10 000 символов
TRANSLATE_MAX_TEXTS = 100


class PipelineError(RuntimeError):
    pass


# ── Модель данных ───────────────────────────────────────────────────────────


@dataclass(slots=True)
class Phrase:
    start: float
    end: float
    source: str
    machine: str = ""
    text: str = ""
    atempo: float = 1.0

    @property
    def slot(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class Cost:
    stt: float = 0.0
    translate: float = 0.0
    tts: float = 0.0

    @property
    def total(self) -> float:
        return self.stt + self.translate + self.tts


@dataclass(slots=True)
class Result:
    phrases: list[Phrase]
    wav_path: Path
    duration: float
    cost: Cost
    sped_up: int = 0
    llm_used: bool = False
    llm_failed_batches: int = 0
    warnings: list[str] = field(default_factory=list)


# ── Object Storage ──────────────────────────────────────────────────────────


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        region_name=config.S3_REGION,
        aws_access_key_id=config.S3_KEY_ID,
        aws_secret_access_key=config.S3_SECRET,
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 5}),
    )


def ensure_bucket() -> None:
    """Создаёт бакет, если его ещё нет. Вызывается один раз при старте бота."""
    client = _s3()
    try:
        client.head_bucket(Bucket=config.S3_BUCKET)
        return
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in {"404", "NoSuchBucket", "403"}:
            raise
        if code == "403":
            raise PipelineError(
                f"Бакет {config.S3_BUCKET} занят другим аккаунтом — придумай другое имя."
            ) from exc
    client.create_bucket(Bucket=config.S3_BUCKET)
    log.info("Создан бакет %s", config.S3_BUCKET)


def _upload(path: Path, key: str) -> str:
    _s3().upload_file(str(path), config.S3_BUCKET, key)
    return f"{config.S3_ENDPOINT.rstrip('/')}/{config.S3_BUCKET}/{key}"


def _delete(key: str) -> None:
    try:
        _s3().delete_object(Bucket=config.S3_BUCKET, Key=key)
    except Exception as exc:  # noqa: BLE001 — уборка не должна ронять задание
        log.warning("Не удалось удалить %s из бакета: %s", key, exc)


# ── HTTP ────────────────────────────────────────────────────────────────────


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Api-Key {config.YC_API_KEY}"}


async def _post_json(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    async with session.post(url, json=payload, headers=_auth_headers()) as resp:
        body = await resp.text()
        if resp.status != 200:
            raise PipelineError(f"{url} вернул {resp.status}: {body[:400]}")
        return await _loads(body)


async def _loads(body: str) -> dict:
    import json

    try:
        return json.loads(body)
    except ValueError as exc:
        raise PipelineError(f"Сервис вернул не JSON: {body[:200]}") from exc


# ── Распознавание ───────────────────────────────────────────────────────────


async def _stt_start(session: aiohttp.ClientSession, uri: str) -> str:
    payload = {
        "config": {
            "specification": {
                "languageCode": config.STT_LANG,
                "model": config.STT_MODEL,
                "profanityFilter": False,
                "audioEncoding": "OGG_OPUS",
                "literature_text": False,
                "audioChannelCount": 1,
                "rawResults": False,
            }
        },
        "audio": {"uri": uri},
    }
    data = await _post_json(session, config.STT_URL, payload)
    op_id = data.get("id")
    if not op_id:
        raise PipelineError(f"SpeechKit не вернул идентификатор операции: {data}")
    return op_id


async def _stt_wait(
    session: aiohttp.ClientSession, op_id: str, progress: Progress | None
) -> list[dict]:
    url = f"{config.OPERATION_URL.rstrip('/')}/{op_id}"
    deadline = time.monotonic() + config.STT_WAIT_MAX_MIN * 60
    delay = 5
    announced = 0.0

    while time.monotonic() < deadline:
        async with session.get(url, headers=_auth_headers()) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise PipelineError(f"Проверка операции вернула {resp.status}: {body[:300]}")
            data = await _loads(body)

        if data.get("done"):
            if "error" in data:
                raise PipelineError(f"Распознавание не удалось: {data['error']}")
            return data.get("response", {}).get("chunks", []) or []

        waited = time.monotonic() - (deadline - config.STT_WAIT_MAX_MIN * 60)
        if progress and waited - announced > 60:
            announced = waited
            await progress(f"Распознаю речь… {int(waited // 60)} мин")
        await asyncio.sleep(delay)
        delay = min(delay + 5, 30)

    raise PipelineError(
        f"Распознавание не закончилось за {config.STT_WAIT_MAX_MIN} минут."
    )


def _parse_time(value: str | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).rstrip("s") or 0.0)


def chunks_to_phrases(chunks: Iterable[dict]) -> list[Phrase]:
    """Собирает фразы из слов с таймкодами: режем по паузе и по длине."""
    phrases: list[Phrase] = []
    words: list[tuple[float, float, str]] = []

    for chunk in chunks:
        alts = chunk.get("alternatives") or []
        if not alts:
            continue
        alt = alts[0]
        raw_words = alt.get("words") or []
        if raw_words:
            for w in raw_words:
                words.append(
                    (_parse_time(w.get("startTime")), _parse_time(w.get("endTime")), w.get("word", ""))
                )
        elif alt.get("text"):
            # запасной путь: слов нет, есть только текст чанка
            phrases.append(Phrase(start=0.0, end=0.0, source=alt["text"].strip()))

    if not words:
        return phrases

    words.sort(key=lambda w: w[0])
    buf: list[tuple[float, float, str]] = []

    def flush() -> None:
        if not buf:
            return
        text = " ".join(w[2] for w in buf).strip()
        if text:
            phrases.append(Phrase(start=buf[0][0], end=buf[-1][1], source=text))
        buf.clear()

    for word in words:
        if buf:
            gap = word[0] - buf[-1][1]
            length = sum(len(w[2]) + 1 for w in buf)
            if gap >= config.PHRASE_GAP or length >= config.PHRASE_MAX_CHARS:
                flush()
        buf.append(word)
    flush()
    return phrases


# ── Перевод ─────────────────────────────────────────────────────────────────


def _batch_by_chars(texts: list[str], max_chars: int, max_items: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    size = 0
    for i, text in enumerate(texts):
        length = len(text)
        if current and (size + length > max_chars or len(current) >= max_items):
            batches.append(current)
            current, size = [], 0
        current.append(i)
        size += length
    if current:
        batches.append(current)
    return batches


async def _translate(session: aiohttp.ClientSession, phrases: list[Phrase]) -> int:
    texts = [p.source for p in phrases]
    total_chars = 0
    for batch in _batch_by_chars(texts, TRANSLATE_MAX_CHARS, TRANSLATE_MAX_TEXTS):
        payload = {
            "folderId": config.YC_FOLDER_ID,
            "texts": [texts[i] for i in batch],
            "targetLanguageCode": "ru",
            "format": "PLAIN_TEXT",
        }
        src = config.STT_LANG.split("-")[0]
        if src and src != "ru":
            payload["sourceLanguageCode"] = src

        data = await _post_json(session, config.TRANSLATE_URL, payload)
        out = data.get("translations") or []
        if len(out) != len(batch):
            raise PipelineError(
                f"Translate вернул {len(out)} строк вместо {len(batch)}."
            )
        for idx, item in zip(batch, out):
            phrases[idx].machine = (item.get("text") or "").strip()
            phrases[idx].text = phrases[idx].machine
        total_chars += sum(len(texts[i]) for i in batch)
    return total_chars


# ── Синтез ──────────────────────────────────────────────────────────────────


async def _tts(session: aiohttp.ClientSession, text: str) -> bytes:
    form = {
        "text": text,
        "lang": "ru-RU",
        "voice": config.TTS_VOICE,
        "speed": str(config.TTS_SPEED),
        "format": "lpcm",
        "sampleRateHertz": str(config.SAMPLE_RATE),
        "folderId": config.YC_FOLDER_ID,
    }
    if config.TTS_EMOTION:
        form["emotion"] = config.TTS_EMOTION

    last_error = ""
    for attempt in range(3):
        try:
            async with session.post(config.TTS_URL, data=form, headers=_auth_headers()) as resp:
                if resp.status == 200:
                    return await resp.read()
                last_error = f"{resp.status}: {(await resp.text())[:200]}"
        except aiohttp.ClientError as exc:
            last_error = str(exc)
        await asyncio.sleep(2 * (attempt + 1))
    raise PipelineError(f"TTS не ответил: {last_error}")


# ── Укладка дорожки ─────────────────────────────────────────────────────────


def _silence(seconds: float) -> bytes:
    samples = max(0, int(round(seconds * config.SAMPLE_RATE)))
    return b"\x00" * (samples * BYTES_PER_SAMPLE)


def _pcm_seconds(raw: bytes) -> float:
    return len(raw) / BYTES_PER_SAMPLE / config.SAMPLE_RATE


async def _build_track(
    session: aiohttp.ClientSession,
    phrases: list[Phrase],
    out_wav: Path,
    total_duration: float,
    progress: Progress | None,
) -> tuple[int, int]:
    """Синтезирует и раскладывает фразы по таймкодам. Пишет потоково.

    Возвращает (сколько символов ушло в синтез, сколько фраз пришлось ускорить).
    """
    tts_chars = 0
    sped_up = 0
    cursor = 0.0
    last_report = time.monotonic()

    with wave.open(str(out_wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(BYTES_PER_SAMPLE)
        out.setframerate(config.SAMPLE_RATE)

        for i, phrase in enumerate(phrases):
            text = (phrase.text or phrase.machine).strip()
            if not text:
                continue

            raw = await _tts(session, text)
            tts_chars += len(text)
            spoken = _pcm_seconds(raw)

            # доступный слот — до начала следующей фразы
            next_start = phrases[i + 1].start if i + 1 < len(phrases) else total_duration
            slot = max(0.0, next_start - max(cursor, phrase.start))

            if slot > 0 and spoken > slot:
                needed = spoken / slot
                if needed >= config.ATEMPO_MIN:
                    tempo = min(needed, config.MAX_ATEMPO)
                    raw = await media.atempo_pcm(raw, tempo)
                    phrase.atempo = tempo
                    spoken = _pcm_seconds(raw)
                    sped_up += 1

            # пауза до начала фразы (паузы автора сохраняем)
            if phrase.start > cursor:
                out.writeframes(_silence(phrase.start - cursor))
                cursor = phrase.start

            out.writeframes(raw)
            cursor += spoken

            if progress and time.monotonic() - last_report > 20:
                last_report = time.monotonic()
                await progress(f"Озвучиваю… {i + 1} из {len(phrases)} фраз")

        if total_duration > cursor:
            out.writeframes(_silence(total_duration - cursor))

    return tts_chars, sped_up


# ── Смета ───────────────────────────────────────────────────────────────────


def estimate_cost(duration: float, src_chars: int, tts_chars: int) -> Cost:
    return Cost(
        stt=math.ceil(duration / 15) * config.STT_RUB_PER_15SEC,
        translate=src_chars / 1_000_000 * config.TRANSLATE_RUB_PER_MCHARS,
        tts=tts_chars / 1_000_000 * config.TTS_RUB_PER_MCHARS,
    )


# ── Пайплайн ────────────────────────────────────────────────────────────────


async def translate_audio(
    audio_path: Path,
    job_dir: Path,
    duration: float,
    progress: Progress | None = None,
) -> Result:
    """Полный проход: аудиофайл на входе → дорожка перевода и фразы на выходе."""
    warnings: list[str] = []

    async def say(text: str) -> None:
        if progress:
            await progress(text)

    await say("Готовлю аудио…")
    ogg = await media.to_ogg_opus(audio_path, job_dir / "stt.ogg")

    key = f"stt/{uuid.uuid4().hex}.ogg"
    await say("Заливаю в хранилище…")
    uri = await asyncio.to_thread(_upload, ogg, key)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await say("Отправляю на распознавание…")
            op_id = await _stt_start(session, uri)
            chunks = await _stt_wait(session, op_id, progress)

            phrases = chunks_to_phrases(chunks)
            if not phrases:
                raise PipelineError("В ролике не нашлось распознаваемой речи.")

            await say(f"Распознано {len(phrases)} фраз. Перевожу…")
            src_chars = await _translate(session, phrases)

            llm_used = False
            failed = 0
            if config.LLM_EDIT:
                await say("Вычитываю перевод…")
                llm_used, failed = await editor.edit_phrases(phrases, progress)
                if failed:
                    warnings.append(
                        f"ИИ-редактор не справился с {failed} блоками — там остался машинный перевод."
                    )

            await say("Озвучиваю…")
            wav_path = job_dir / "dub.wav"
            tts_chars, sped_up = await _build_track(
                session, phrases, wav_path, duration, progress
            )
    finally:
        await asyncio.to_thread(_delete, key)

    return Result(
        phrases=phrases,
        wav_path=wav_path,
        duration=duration,
        cost=estimate_cost(duration, src_chars, tts_chars),
        sped_up=sped_up,
        llm_used=llm_used,
        llm_failed_batches=failed,
        warnings=warnings,
    )
