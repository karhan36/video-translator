"""Работа с медиа: скачивание через yt-dlp, конвертация и сборка через ffmpeg."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import config

log = logging.getLogger(__name__)


class MediaError(RuntimeError):
    pass


@dataclass(slots=True)
class Probe:
    duration: float
    width: int = 0
    height: int = 0


async def _run(cmd: list[str], timeout: int = 3600) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise MediaError(f"Команда не уложилась в {timeout} с: {cmd[0]}")
    return proc.returncode, out, err


def _check_tools() -> None:
    for tool in ("ffmpeg", "ffprobe", "yt-dlp"):
        if shutil.which(tool) is None:
            raise MediaError(f"Не найден {tool}. Установи его на сервере.")


# ── ffprobe ─────────────────────────────────────────────────────────────────


async def probe(path: Path) -> Probe:
    code, out, err = await _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=300,
    )
    if code != 0:
        raise MediaError(f"ffprobe не смог прочитать файл: {err.decode(errors='replace')[:400]}")

    data = json.loads(out or b"{}")
    duration = float(data.get("format", {}).get("duration") or 0.0)
    width = height = 0
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            if not duration:
                duration = float(stream.get("duration") or 0.0)
            break
    if duration <= 0:
        raise MediaError("Не удалось определить длительность файла.")
    return Probe(duration=duration, width=width, height=height)


# ── yt-dlp ──────────────────────────────────────────────────────────────────


# Наборы клиентов плеера, по которым идёт перебор при отказе YouTube.
# Проверено 6 сентября 2026 на Timeweb ams-1: default решает JS-челлендж через
# Deno, android_vr работает без него. Проверка «вы не робот» у YouTube плавающая:
# один и тот же ролик минуту назад открывался, а сейчас требует подтверждения,
# поэтому важен не «правильный» клиент, а повтор другим набором.
# 8 сентября 2026: default и android_vr ловят бот-проверку, поэтому первыми идут
# mweb и web_embedded. Формат отдают только с пакетом yt-dlp-ejs (решатель n-челленджа).
YTDLP_CLIENTS = (
    "mweb,web_embedded",
    "web_embedded",
    "default,android_vr",
    "android_vr",
    "default",
    "tv,web_safari",
)

# Признаки того, что отказ временный и стоит повторить другим клиентом.
_RETRYABLE = (
    "sign in to confirm",
    "not a bot",
    "page needs to be reloaded",
    "requested format is not available",
    "http error 429",
    "unable to extract",
    "failed to extract any player response",
)


def _ytdlp_base(clients: str = YTDLP_CLIENTS[0]) -> list[str]:
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--socket-timeout",
        "30",
        "--extractor-args",
        f"youtube:player_client={clients}",
    ]
    if config.YTDLP_COOKIES:
        cmd += ["--cookies", config.YTDLP_COOKIES]
    if config.YTDLP_PROXY:
        cmd += ["--proxy", config.YTDLP_PROXY]
    return cmd


def is_retryable_error(stderr: str) -> bool:
    """Отказ YouTube, который лечится повтором с другим клиентом плеера."""
    low = stderr.lower()
    return any(mark in low for mark in _RETRYABLE)


async def _run_ytdlp(
    args: list[str], url: str, timeout: int, pause: float = 3.0
) -> tuple[bytes, bytes]:
    """Запускает yt-dlp, перебирая клиентов плеера, пока YouTube не отдаст ролик.

    Возвращает stdout и stderr удачной попытки. Если все наборы упали,
    поднимает MediaError с текстом последней ошибки.
    """
    _check_tools()
    last_err = b""
    for attempt, clients in enumerate(YTDLP_CLIENTS):
        code, out, err = await _run(_ytdlp_base(clients) + args + [url], timeout=timeout)
        if code == 0:
            if attempt:
                log.info("yt-dlp: сработал клиент %s с попытки %d", clients, attempt + 1)
            return out, err
        last_err = err
        text = err.decode(errors="replace")
        if not is_retryable_error(text):
            break
        log.warning("yt-dlp: клиент %s отказал, пробую следующий", clients)
        if pause:
            await asyncio.sleep(pause)
    raise MediaError(last_err.decode(errors="replace")[-400:] or "yt-dlp не дал ответа.")


async def fetch_info(url: str) -> dict:
    """Метаданные ролика без скачивания."""
    try:
        out, _ = await _run_ytdlp(["--dump-single-json"], url, timeout=180)
    except MediaError as exc:
        raise MediaError(
            "yt-dlp не смог открыть ссылку. Проверь, что ролик доступен.\n" + str(exc)
        ) from exc
    return json.loads(out or b"{}")


# ── Имя ролика ──────────────────────────────────────────────────────────────

# «Мусорным» считаем имя без пробелов, целиком из hex или из случайной
# мешанины букв, цифр и разделителей: так выглядит имя файла из прямой ссылки.
_HEX_RE = re.compile(r"^[0-9a-fA-F]{8,}$")
_SLUG_RE = re.compile(r"^[0-9A-Za-z_\-]{12,}$")


def is_junk_title(title: str) -> bool:
    """Имя выглядит как хеш или техническая строка, а не как название ролика."""
    name = (title or "").strip()
    if not name:
        return True
    if name.startswith(("http://", "https://")):
        return True
    stem = re.sub(r"\.[A-Za-z0-9]{2,4}$", "", name)  # отрезаем расширение
    if _HEX_RE.match(stem):
        return True
    if _SLUG_RE.match(stem) and not re.search(r"[А-Яа-яЁё]", stem):
        # длинная строка без пробелов: если букв мало или цифр много — хеш
        digits = sum(c.isdigit() for c in stem)
        return digits >= 4 or len(stem) >= 20
    return False


def title_from_tags(tags: dict | None) -> str:
    """Название из тегов самого файла (ID3 и аналоги)."""
    tags = {str(k).lower(): v for k, v in (tags or {}).items()}
    title = str(tags.get("title") or "").strip()
    if not title or is_junk_title(title):
        return ""
    artist = str(tags.get("artist") or tags.get("album_artist") or "").strip()
    return f"{artist} — {title}" if artist and artist.lower() not in title.lower() else title


def fallback_title(url: str) -> str:
    """Запасное имя, когда названия нет нигде: домен и дата."""
    host = urlparse(url).netloc.replace("www.", "") or "видео"
    return f"Перевод {host} {datetime.now():%Y-%m-%d}"


def nice_title(info: dict, url: str) -> str:
    """Человекочитаемое название ролика из метаданных yt-dlp.

    Для прямых ссылок yt-dlp подставляет имя файла из адреса — часто хеш.
    Тогда перебираем другие поля, а в конце отдаём пустую строку: вызывающий
    код попробует теги скачанного файла и только потом fallback_title.
    """
    for key in ("track", "title", "fulltitle", "alt_title"):
        value = str(info.get(key) or "").strip()
        if value and not is_junk_title(value):
            return value
    return ""


async def file_tags(path: Path) -> dict:
    """Теги медиафайла (format.tags) через ffprobe. Ошибки не роняют задание."""
    try:
        code, out, _ = await _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ],
            timeout=120,
        )
        if code != 0:
            return {}
        return json.loads(out or b"{}").get("format", {}).get("tags", {}) or {}
    except Exception:  # noqa: BLE001
        return {}


async def download_audio(url: str, out_dir: Path) -> Path:
    """Скачивает только звуковую дорожку."""
    target = out_dir / "source_audio.%(ext)s"
    try:
        await _run_ytdlp(
            ["-f", "bestaudio/best", "--no-part", "-o", str(target)],
            url,
            timeout=7200,
        )
    except MediaError as exc:
        raise MediaError("Не получилось скачать аудио.\n" + str(exc)) from exc
    files = sorted(out_dir.glob("source_audio.*"))
    if not files:
        raise MediaError("yt-dlp отработал, но файла нет.")
    return files[0]


async def download_video(url: str, out_dir: Path) -> Path:
    """Скачивает видео не выше VIDEO_HEIGHT."""
    height = config.VIDEO_HEIGHT
    fmt = (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={height}]+bestaudio/best[height<={height}]/best"
    )
    target = out_dir / "source_video.%(ext)s"
    try:
        await _run_ytdlp(
            ["-f", fmt, "--merge-output-format", "mp4", "--no-part", "-o", str(target)],
            url,
            timeout=10800,
        )
    except MediaError as exc:
        raise MediaError("Не получилось скачать видео.\n" + str(exc)) from exc
    files = sorted(out_dir.glob("source_video.*"))
    if not files:
        raise MediaError("yt-dlp отработал, но видеофайла нет.")
    return files[0]


# ── ffmpeg ──────────────────────────────────────────────────────────────────


async def to_ogg_opus(src: Path, dst: Path) -> Path:
    """OGG_OPUS, моно, 16 кГц — формат для асинхронного распознавания."""
    _check_tools()
    code, _, err = await _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            str(dst),
        ],
        timeout=7200,
    )
    if code != 0:
        raise MediaError("ffmpeg не смог сконвертировать аудио:\n" + err.decode(errors="replace")[-400:])
    return dst


async def wav_to_mp3(src: Path, dst: Path) -> Path:
    _check_tools()
    code, _, err = await _run(
        ["ffmpeg", "-y", "-i", str(src), "-c:a", "libmp3lame", "-b:a", config.AUDIO_BITRATE, str(dst)],
        timeout=3600,
    )
    if code != 0:
        raise MediaError("ffmpeg не смог собрать mp3:\n" + err.decode(errors="replace")[-400:])
    return dst


async def mix_video(video: Path, dub_wav: Path, dst: Path) -> Path:
    """Дорожка перевода поверх приглушённого оригинала."""
    _check_tools()
    filt = (
        f"[0:a]volume={config.ORIG_VOLUME}[orig];"
        f"[1:a]volume=1.0[dub];"
        f"[orig][dub]amix=inputs=2:duration=first:dropout_transition=0,"
        f"alimiter=limit=0.95[out]"
    )
    code, _, err = await _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(dub_wav),
            "-filter_complex",
            filt,
            "-map",
            "0:v:0",
            "-map",
            "[out]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            config.AUDIO_BITRATE,
            "-movflags",
            "+faststart",
            str(dst),
        ],
        timeout=10800,
    )
    if code != 0:
        # оригинал может быть без звуковой дорожки — тогда просто подставляем перевод
        log.warning("amix не прошёл, кладу дорожку перевода поверх немого видео")
        code2, _, err2 = await _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-i",
                str(dub_wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                config.AUDIO_BITRATE,
                "-shortest",
                "-movflags",
                "+faststart",
                str(dst),
            ],
            timeout=10800,
        )
        if code2 != 0:
            raise MediaError(
                "ffmpeg не смог собрать видео:\n"
                + err.decode(errors="replace")[-300:]
                + "\n"
                + err2.decode(errors="replace")[-300:]
            )
    return dst


async def atempo_pcm(raw: bytes, tempo: float) -> bytes:
    """Ускоряет кусок сырого PCM без изменения тона."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(config.SAMPLE_RATE),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-filter:a",
        f"atempo={tempo:.4f}",
        "-f",
        "s16le",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(raw)
    if proc.returncode != 0 or not out:
        log.warning("atempo не отработал (%s), оставляю фразу как есть", err[-200:])
        return raw
    return out
