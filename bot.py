"""Телеграм-бот перевода видео. aiogram 3, движок — Yandex SpeechKit."""

from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

import config
import media
import speechkit
import textout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("bot")

URL_RE = re.compile(r"https?://\S+")

dp = Dispatcher()
job_lock = asyncio.Lock()


# ── Состояние заданий ───────────────────────────────────────────────────────


@dataclass
class Job:
    job_id: str
    chat_id: int
    source_url: str = ""
    title: str = ""
    dir: Path = field(default_factory=Path)
    result: speechkit.Result | None = None
    audio_only: bool = False
    local_video: Path | None = None  # видео, присланное файлом: качать неоткуда
    mp3: Path | None = None
    mp4: Path | None = None
    srt: Path | None = None
    txt: Path | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


JOBS: dict[str, Job] = {}


# ── Вспомогательное ─────────────────────────────────────────────────────────


def allowed(user_id: int | None) -> bool:
    return user_id is not None and user_id in config.ALLOWED_USER_IDS


def hhmmss(seconds: float) -> str:
    total = int(round(seconds))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def keyboard(job: Job) -> InlineKeyboardMarkup:
    row = []
    if not job.audio_only:
        row.append(InlineKeyboardButton(text="🎬 Видео", callback_data=f"get:video:{job.job_id}"))
    row.append(InlineKeyboardButton(text="🎧 Аудио", callback_data=f"get:audio:{job.job_id}"))
    row.append(InlineKeyboardButton(text="📄 Текст", callback_data=f"get:text:{job.job_id}"))
    return InlineKeyboardMarkup(inline_keyboard=[row])


def cost_report(job: Job) -> str:
    result = job.result
    assert result is not None
    cost = result.cost
    lines = [
        f"<b>{html.escape(job.title or 'Готово')}</b>",
        f"Длительность: {hhmmss(result.duration)} · фраз: {len(result.phrases)}",
        "",
        "<b>Примерная стоимость</b>",
        f"  распознавание — {cost.stt:.2f} ₽",
        f"  перевод — {cost.translate:.2f} ₽",
        f"  синтез — {cost.tts:.2f} ₽",
        f"  <b>итого ≈ {cost.total:.2f} ₽</b>",
    ]
    if result.sped_up:
        share = result.sped_up / max(1, len(result.phrases)) * 100
        lines.append("")
        lines.append(f"Ускорено фраз: {result.sped_up} ({share:.0f}%)")
    for warn in result.warnings:
        lines.append("")
        lines.append("⚠️ " + html.escape(warn))
    lines.append("")
    lines.append("Файлы удалятся через %d мин." % config.CLEANUP_MIN)
    return "\n".join(lines)


class Reporter:
    """Один редактируемый статус вместо потока сообщений."""

    def __init__(self, message: Message) -> None:
        self.message = message
        self.last_text = ""
        self.last_time = 0.0

    async def __call__(self, text: str) -> None:
        now = time.monotonic()
        if text == self.last_text or now - self.last_time < 3:
            return
        self.last_text, self.last_time = text, now
        with contextlib.suppress(Exception):
            await self.message.edit_text(text)


async def cleanup_later(job: Job) -> None:
    await asyncio.sleep(config.CLEANUP_MIN * 60)
    shutil.rmtree(job.dir, ignore_errors=True)
    JOBS.pop(job.job_id, None)
    log.info("Убрал файлы задания %s", job.job_id)


def sweep_leftovers() -> None:
    """Хвосты прошлых запусков — при старте."""
    if not config.WORK_DIR.exists():
        return
    for path in config.WORK_DIR.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    log.info("Подчистил рабочую папку")


# ── Команды ─────────────────────────────────────────────────────────────────


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not allowed(message.from_user.id if message.from_user else None):
        return
    await message.answer(
        "Пришли ссылку на ролик — переведу речь на русский, озвучу и уложу "
        "по таймкодам оригинала.\n\n"
        "Можно прислать файлом: видео (mp4, mkv, mov, webm), аудио, голосовое. "
        "Для видео дорожка перевода ляжет поверх оригинала.\n\n"
        f"Лимит длины: {config.MAX_DURATION_MIN} мин. Один ролик в работе за раз."
    )


@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    if not allowed(message.from_user.id if message.from_user else None):
        return
    busy = "занят переводом" if job_lock.locked() else "свободен"
    await message.answer(
        f"Бот {busy}. Готовых заданий в памяти: {len(JOBS)}.\n"
        f"Язык оригинала: {config.STT_LANG} · голос: {config.TTS_VOICE} · "
        f"ИИ-редактор: {'вкл' if config.LLM_EDIT else 'выкл'}"
    )


# ── Приём работы ────────────────────────────────────────────────────────────


@dp.message(F.audio | F.voice | F.video | F.video_note | F.document)
async def on_media(message: Message) -> None:
    """Приём файлов: аудио и видео. Для видео дорожка перевода кладётся поверх него."""
    if not allowed(message.from_user.id if message.from_user else None):
        return

    file_obj = (
        message.audio or message.voice or message.video or message.video_note or message.document
    )
    is_video = bool(message.video or message.video_note)

    if message.document:
        mime = (message.document.mime_type or "").lower()
        name = (message.document.file_name or "").lower()
        if mime.startswith("video") or name.endswith((".mp4", ".mkv", ".mov", ".webm", ".avi")):
            is_video = True
        elif not mime.startswith("audio"):
            await message.answer(
                "Не понял формат. Пришли видео, аудио, голосовое или ссылку на ролик."
            )
            return

    if job_lock.locked():
        await message.answer("Сейчас в работе другой ролик. Дождись, пожалуйста.")
        return

    async with job_lock:
        job = Job(
            job_id=uuid.uuid4().hex[:10], chat_id=message.chat.id, audio_only=not is_video
        )
        job.dir = config.WORK_DIR / job.job_id
        job.dir.mkdir(parents=True, exist_ok=True)
        job.title = getattr(file_obj, "file_name", None) or (
            "Видеозапись" if is_video else "Аудиозапись"
        )

        status = await message.answer("Скачиваю файл…")
        report = Reporter(status)
        try:
            src = job.dir / ("source_video.mp4" if is_video else "source_input")
            await message.bot.download(file_obj, destination=src)
            if is_video:
                job.local_video = src
            probe = await media.probe(src)
            if probe.duration > config.MAX_DURATION_MIN * 60:
                await status.edit_text(
                    f"Слишком длинно: {hhmmss(probe.duration)}. "
                    f"Лимит — {config.MAX_DURATION_MIN} мин."
                )
                shutil.rmtree(job.dir, ignore_errors=True)
                return

            job.result = await speechkit.translate_audio(src, job.dir, probe.duration, report)
        except Exception as exc:  # noqa: BLE001
            log.exception("Задание %s упало", job.job_id)
            await status.edit_text("Не получилось.\n\n" + html.escape(str(exc))[:1500])
            shutil.rmtree(job.dir, ignore_errors=True)
            return

        JOBS[job.job_id] = job
        await status.edit_text(cost_report(job), reply_markup=keyboard(job))
        asyncio.create_task(cleanup_later(job))


@dp.message(F.text.regexp(URL_RE))
async def on_link(message: Message) -> None:
    if not allowed(message.from_user.id if message.from_user else None):
        return

    match = URL_RE.search(message.text or "")
    if not match:
        return
    url = match.group(0)

    if job_lock.locked():
        await message.answer("Сейчас в работе другой ролик. Дождись, пожалуйста.")
        return

    async with job_lock:
        job = Job(job_id=uuid.uuid4().hex[:10], chat_id=message.chat.id, source_url=url)
        job.dir = config.WORK_DIR / job.job_id
        job.dir.mkdir(parents=True, exist_ok=True)

        status = await message.answer("Смотрю ролик…")
        report = Reporter(status)
        try:
            info = await media.fetch_info(url)
            job.title = info.get("title") or url
            duration = float(info.get("duration") or 0)
            if duration and duration > config.MAX_DURATION_MIN * 60:
                await status.edit_text(
                    f"«{html.escape(job.title)}» идёт {hhmmss(duration)}. "
                    f"Лимит — {config.MAX_DURATION_MIN} мин."
                )
                shutil.rmtree(job.dir, ignore_errors=True)
                return

            await report(f"Качаю звук: {job.title[:60]}")
            audio = await media.download_audio(url, job.dir)
            if not duration:
                duration = (await media.probe(audio)).duration

            job.result = await speechkit.translate_audio(audio, job.dir, duration, report)
        except Exception as exc:  # noqa: BLE001
            log.exception("Задание %s упало", job.job_id)
            await status.edit_text("Не получилось.\n\n" + html.escape(str(exc))[:1500])
            shutil.rmtree(job.dir, ignore_errors=True)
            return

        JOBS[job.job_id] = job
        await status.edit_text(cost_report(job), reply_markup=keyboard(job))
        asyncio.create_task(cleanup_later(job))


@dp.message(F.text)
async def on_other_text(message: Message) -> None:
    if not allowed(message.from_user.id if message.from_user else None):
        return
    await message.answer("Пришли ссылку на ролик или аудиофайл.")


# ── Выдача результатов ──────────────────────────────────────────────────────


@dp.callback_query(F.data.startswith("get:"))
async def on_get(call: CallbackQuery) -> None:
    if not allowed(call.from_user.id if call.from_user else None):
        await call.answer()
        return

    _, kind, job_id = call.data.split(":", 2)
    job = JOBS.get(job_id)
    if not job or not job.result:
        await call.answer("Файлы этого задания уже удалены.", show_alert=True)
        return

    await call.answer()
    async with job.lock:
        try:
            if kind == "audio":
                await send_audio(call.message, job)
            elif kind == "text":
                await send_text(call.message, job)
            elif kind == "video":
                await send_video(call.message, job)
        except Exception as exc:  # noqa: BLE001
            log.exception("Отдача %s задания %s упала", kind, job_id)
            await call.message.answer("Не смог отдать файл.\n\n" + html.escape(str(exc))[:1000])


async def send_audio(message: Message, job: Job) -> None:
    assert job.result is not None
    if not job.mp3 or not job.mp3.exists():
        note = await message.answer("Собираю mp3…")
        job.mp3 = await media.wav_to_mp3(job.result.wav_path, job.dir / "dub.mp3")
        with contextlib.suppress(Exception):
            await note.delete()
    await message.answer_audio(
        FSInputFile(job.mp3, filename=f"{safe_name(job.title)}.mp3"),
        title=job.title[:64],
        duration=int(job.result.duration),
    )


async def send_text(message: Message, job: Job) -> None:
    assert job.result is not None
    if not job.srt or not job.srt.exists():
        job.srt = textout.write_srt(job.result.phrases, job.dir / "subs.srt")
    if not job.txt or not job.txt.exists():
        job.txt = textout.write_txt(job.result.phrases, job.dir / "text.txt", job.title)
    await message.answer_document(
        FSInputFile(job.srt, filename=f"{safe_name(job.title)}.srt")
    )
    await message.answer_document(
        FSInputFile(job.txt, filename=f"{safe_name(job.title)}.txt")
    )


async def send_video(message: Message, job: Job) -> None:
    assert job.result is not None
    if job.audio_only:
        await message.answer("Для аудиофайла видео нет.")
        return

    if not job.mp4 or not job.mp4.exists():
        if job.local_video and job.local_video.exists():
            # видео прислали файлом — качать нечего, сразу накладываем дорожку
            note = await message.answer("Накладываю дорожку на видео…")
            video = job.local_video
        elif job.source_url:
            note = await message.answer("Качаю видео и накладываю дорожку. Это дольше всего…")
            video = await media.download_video(job.source_url, job.dir)
        else:
            await message.answer("Исходного видео нет.")
            return
        job.mp4 = await media.mix_video(video, job.result.wav_path, job.dir / "dubbed.mp4")
        with contextlib.suppress(Exception):
            await note.delete()

    probe = await media.probe(job.mp4)
    await message.answer_video(
        FSInputFile(job.mp4, filename=f"{safe_name(job.title)}.mp4"),
        width=probe.width or None,
        height=probe.height or None,
        duration=int(probe.duration),
        supports_streaming=True,
    )


def safe_name(title: str) -> str:
    name = re.sub(r"[^\w\s.-]", "", title, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", "_", name)
    return (name or "translation")[:80]


# ── Запуск ──────────────────────────────────────────────────────────────────


async def main() -> None:
    missing = config.missing_settings()
    if missing:
        raise SystemExit("В .env не заполнено: " + ", ".join(missing))

    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    sweep_leftovers()

    await asyncio.to_thread(speechkit.ensure_bucket)

    session = None
    if config.TG_API_BASE:
        session = AiohttpSession(
            api=TelegramAPIServer.from_base(config.TG_API_BASE, is_local=True)
        )
        log.info("Работаю через локальный Bot API server: %s", config.TG_API_BASE)

    bot = Bot(
        token=config.BOT_TOKEN,
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    me = await bot.get_me()
    log.info("Запущен как @%s, доступ у %s", me.username, sorted(config.ALLOWED_USER_IDS))

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
