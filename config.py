"""Настройки бота. Всё читается из .env, значения по умолчанию — рабочие."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "да"}


# ── Telegram ────────────────────────────────────────────────────────────────
BOT_TOKEN = _str("BOT_TOKEN")
TG_API_BASE = _str("TG_API_BASE")  # http://127.0.0.1:8081 для локального Bot API server
ALLOWED_USER_IDS = {
    int(x) for x in _str("ALLOWED_USER_IDS").replace(";", ",").split(",") if x.strip()
}

# ── Yandex Cloud ────────────────────────────────────────────────────────────
YC_API_KEY = _str("YC_API_KEY")
YC_FOLDER_ID = _str("YC_FOLDER_ID")

S3_ENDPOINT = _str("S3_ENDPOINT", "https://storage.yandexcloud.net")
S3_REGION = _str("S3_REGION", "ru-central1")
S3_KEY_ID = _str("S3_KEY_ID")
S3_SECRET = _str("S3_SECRET")
S3_BUCKET = _str("S3_BUCKET")

STT_URL = _str(
    "STT_URL",
    "https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize",
)
OPERATION_URL = _str("OPERATION_URL", "https://operation.api.cloud.yandex.net/operations")
TRANSLATE_URL = _str(
    "TRANSLATE_URL", "https://translate.api.cloud.yandex.net/translate/v2/translate"
)
TTS_URL = _str("TTS_URL", "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize")

# ── Распознавание и синтез ──────────────────────────────────────────────────
STT_LANG = _str("STT_LANG", "en-US")
STT_MODEL = _str("STT_MODEL", "general")
TTS_VOICE = _str("TTS_VOICE", "filipp")
TTS_SPEED = _float("TTS_SPEED", 1.0)
TTS_EMOTION = _str("TTS_EMOTION", "neutral")

SAMPLE_RATE = 48000  # частота дорожки перевода, Гц (совпадает с lpcm по умолчанию)

# ── Нарезка на фразы ────────────────────────────────────────────────────────
PHRASE_GAP = _float("PHRASE_GAP", 0.7)  # пауза между словами, с которой режем фразу
PHRASE_MAX_CHARS = _int("PHRASE_MAX_CHARS", 220)

# ── Укладка озвучки ─────────────────────────────────────────────────────────
MAX_ATEMPO = _float("MAX_ATEMPO", 1.5)  # потолок ускорения, выше — голос ломается
ATEMPO_MIN = _float("ATEMPO_MIN", 1.12)  # ниже порога не ускоряем вовсе
CHARS_PER_SEC = _float("CHARS_PER_SEC", 15.0)  # бюджет символов на секунду речи

# ── ИИ-редактор ─────────────────────────────────────────────────────────────
LLM_EDIT = _bool("LLM_EDIT", True)
LLM_BATCH = _int("LLM_BATCH", 20)
LLM_API_BASE = _str("LLM_API_BASE", "https://ai.api.cloud.yandex.net/v1")
LLM_MODEL = _str("LLM_MODEL", "yandexgpt/latest")
LLM_API_KEY = _str("LLM_API_KEY") or YC_API_KEY
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.3)
LLM_TIMEOUT = _int("LLM_TIMEOUT", 120)
GLOSSARY_FILE = _str("GLOSSARY_FILE", str(BASE_DIR / "glossary.txt"))

# ── Ограничения и файлы ─────────────────────────────────────────────────────
MAX_DURATION_MIN = _int("MAX_DURATION_MIN", 180)
STT_WAIT_MAX_MIN = _int("STT_WAIT_MAX_MIN", 300)  # сколько ждём распознавание
CLEANUP_MIN = _int("CLEANUP_MIN", 30)
WORK_DIR = Path(_str("WORK_DIR", str(BASE_DIR / "work")))

VIDEO_HEIGHT = _int("VIDEO_HEIGHT", 720)
ORIG_VOLUME = _float("ORIG_VOLUME", 0.15)  # громкость оригинала под озвучкой
AUDIO_BITRATE = _str("AUDIO_BITRATE", "192k")

YTDLP_COOKIES = _str("YTDLP_COOKIES")  # путь к cookies.txt, если нужен
YTDLP_PROXY = _str("YTDLP_PROXY")

# ── Разбивка текста на абзацы ───────────────────────────────────────────────
PARA_GAP = _float("PARA_GAP", 2.0)
PARA_MAX_CHARS = _int("PARA_MAX_CHARS", 900)

# ── Тарифы Yandex Cloud, ₽ с НДС ────────────────────────────────────────────
STT_RUB_PER_15SEC = _float("STT_RUB_PER_15SEC", 0.1515)
TRANSLATE_RUB_PER_MCHARS = _float("TRANSLATE_RUB_PER_MCHARS", 500.4)
TTS_RUB_PER_MCHARS = _float("TTS_RUB_PER_MCHARS", 1342.0)


def missing_settings() -> list[str]:
    """Что обязательно должно быть заполнено в .env."""
    required = {
        "BOT_TOKEN": BOT_TOKEN,
        "ALLOWED_USER_IDS": ALLOWED_USER_IDS,
        "YC_API_KEY": YC_API_KEY,
        "YC_FOLDER_ID": YC_FOLDER_ID,
        "S3_KEY_ID": S3_KEY_ID,
        "S3_SECRET": S3_SECRET,
        "S3_BUCKET": S3_BUCKET,
    }
    return [name for name, value in required.items() if not value]
