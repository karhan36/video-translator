"""Работа с медиа: скачивание через yt-dlp, конвертация и сборка через ffmpeg."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

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


def _ytdlp_base() -> list[str]:
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
        # обходы блокировок YouTube: пробуем несколько клиентов плеера
        "--extractor-args",
        "youtube:player_client=tv,ios,web",
    ]
    if config.YTDLP_COOKIES:
        cmd += ["--cookies", config.YTDLP_COOKIES]
    if config.YTDLP_PROXY:
        cmd += ["--proxy", config.YTDLP_PROXY]
    return cmd


async def fetch_info(url: str) -> dict:
    """Метаданные ролика без скачивания."""
    _check_tools()
    code, out, err = await _run(_ytdlp_base() + ["--dump-single-json", url], timeout=180)
    if code != 0:
        raise MediaError(
            "yt-dlp не смог открыть ссылку. Проверь, что ролик доступен.\n"
            + err.decode(errors="replace")[-400:]
        )
    return json.loads(out or b"{}")


async def download_audio(url: str, out_dir: Path) -> Path:
    """Скачивает только звуковую дорожку."""
    _check_tools()
    target = out_dir / "source_audio.%(ext)s"
    code, _, err = await _run(
        _ytdlp_base()
        + ["-f", "bestaudio/best", "--no-part", "-o", str(target), url],
        timeout=7200,
    )
    if code != 0:
        raise MediaError(
            "Не получилось скачать аудио.\n" + err.decode(errors="replace")[-400:]
        )
    files = sorted(out_dir.glob("source_audio.*"))
    if not files:
        raise MediaError("yt-dlp отработал, но файла нет.")
    return files[0]


async def download_video(url: str, out_dir: Path) -> Path:
    """Скачивает видео не выше VIDEO_HEIGHT."""
    _check_tools()
    height = config.VIDEO_HEIGHT
    fmt = (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={height}]+bestaudio/best[height<={height}]/best"
    )
    target = out_dir / "source_video.%(ext)s"
    code, _, err = await _run(
        _ytdlp_base()
        + ["-f", fmt, "--merge-output-format", "mp4", "--no-part", "-o", str(target), url],
        timeout=10800,
    )
    if code != 0:
        raise MediaError(
            "Не получилось скачать видео.\n" + err.decode(errors="replace")[-400:]
        )
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
