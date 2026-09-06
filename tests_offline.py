"""Офлайн-проверки логики: нарезка фраз, укладка по таймкодам, srt/txt, смета.

Внешние сервисы не дёргаются — TTS подменён синтетическим PCM.
Запуск: python3 tests_offline.py
"""

from __future__ import annotations

import asyncio
import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path

os.environ.setdefault("BOT_TOKEN", "x")
os.environ.setdefault("ALLOWED_USER_IDS", "1")
os.environ.setdefault("YC_API_KEY", "x")
os.environ.setdefault("YC_FOLDER_ID", "x")
os.environ.setdefault("S3_KEY_ID", "x")
os.environ.setdefault("S3_SECRET", "x")
os.environ.setdefault("S3_BUCKET", "x")

import config  # noqa: E402
import speechkit  # noqa: E402
import textout  # noqa: E402
from speechkit import Phrase  # noqa: E402

FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILED.append(name)


def approx(a: float, b: float, tol: float = 0.05) -> bool:
    return abs(a - b) <= tol


# ── 1. Нарезка чанков на фразы ──────────────────────────────────────────────


def test_phrases() -> None:
    def word(w, s, e):
        return {"word": w, "startTime": f"{s}s", "endTime": f"{e}s"}

    chunks = [
        {
            "alternatives": [
                {
                    "text": "hello world and then a long pause here",
                    "words": [
                        word("hello", 0.0, 0.4),
                        word("world", 0.45, 0.9),
                        # пауза 1.5 с — здесь должна быть граница фразы
                        word("and", 2.4, 2.6),
                        word("then", 2.65, 2.9),
                        word("a", 2.95, 3.0),
                        word("long", 3.05, 3.4),
                    ],
                }
            ]
        }
    ]
    phrases = speechkit.chunks_to_phrases(chunks)
    check("нарезка: две фразы по паузе", len(phrases) == 2, f"получилось {len(phrases)}")
    check("нарезка: первая фраза", phrases[0].source == "hello world", phrases[0].source)
    check("нарезка: тайминг первой", approx(phrases[0].start, 0.0) and approx(phrases[0].end, 0.9))
    check("нарезка: тайминг второй", approx(phrases[1].start, 2.4) and approx(phrases[1].end, 3.4))

    # чанк без слов не должен теряться
    fallback = speechkit.chunks_to_phrases([{"alternatives": [{"text": "no words here"}]}])
    check("нарезка: чанк без слов не теряется", len(fallback) == 1)

    # пустой вход
    check("нарезка: пустой вход", speechkit.chunks_to_phrases([]) == [])


# ── 2. Батчинг для Translate ────────────────────────────────────────────────


def test_batching() -> None:
    texts = ["x" * 4000, "y" * 4000, "z" * 4000]
    batches = speechkit._batch_by_chars(texts, 9000, 100)
    check("батчинг: режет по символам", len(batches) == 2, str([len(b) for b in batches]))
    check("батчинг: порядок сохранён", [i for b in batches for i in b] == [0, 1, 2])

    many = ["a"] * 250
    batches = speechkit._batch_by_chars(many, 9000, 100)
    check("батчинг: режет по числу строк", all(len(b) <= 100 for b in batches) and len(batches) == 3)


# ── 3. Укладка дорожки ──────────────────────────────────────────────────────


def fake_pcm(seconds: float) -> bytes:
    """Тихий, но не нулевой сигнал — чтобы отличать речь от тишины."""
    n = int(seconds * config.SAMPLE_RATE)
    return struct.pack("<%dh" % n, *([1000] * n))


async def run_layout(phrases: list[Phrase], total: float, spoken: dict[int, float]):
    """Подменяем TTS: i-я фраза озвучивается spoken[i] секунд."""
    order = {id(p): i for i, p in enumerate(phrases)}
    calls = {"n": 0}

    async def fake_tts(session, text):
        i = calls["n"]
        calls["n"] += 1
        return fake_pcm(spoken[i])

    original = speechkit._tts
    speechkit._tts = fake_tts
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dub.wav"
            chars, sped = await speechkit._build_track(None, phrases, out, total, None)
            with wave.open(str(out), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                raw = w.readframes(frames)
            return frames / rate, chars, sped, raw, rate
    finally:
        speechkit._tts = original


def loudness_at(raw: bytes, rate: int, second: float) -> int:
    idx = int(second * rate) * 2
    if idx + 2 > len(raw):
        return -1
    return abs(struct.unpack_from("<h", raw, idx)[0])


def test_layout_fits() -> None:
    """Фразы влезают в слоты — паузы автора сохраняются, ускорения нет."""
    phrases = [
        Phrase(start=1.0, end=2.0, source="a", text="раз"),
        Phrase(start=5.0, end=6.0, source="b", text="два"),
    ]
    dur, chars, sped, raw, rate = asyncio.run(
        run_layout(phrases, total=8.0, spoken={0: 1.0, 1: 1.0})
    )
    check("укладка: длина дорожки равна длине ролика", approx(dur, 8.0, 0.1), f"{dur:.2f} с")
    check("укладка: ускорений нет", sped == 0)
    check("укладка: символы посчитаны", chars == len("раз") + len("два"), str(chars))
    check("укладка: тишина до первой фразы", loudness_at(raw, rate, 0.5) == 0)
    check("укладка: речь на 1.5 с", loudness_at(raw, rate, 1.5) > 0)
    check("укладка: пауза автора на 3 с сохранена", loudness_at(raw, rate, 3.0) == 0)
    check("укладка: речь на 5.5 с", loudness_at(raw, rate, 5.5) > 0)


def test_layout_speedup() -> None:
    """Фраза длиннее слота и превышение выше порога — ускоряем."""
    phrases = [
        Phrase(start=0.0, end=2.0, source="a", text="длинная фраза"),
        Phrase(start=2.0, end=4.0, source="b", text="вторая"),
    ]
    dur, _, sped, _, _ = asyncio.run(run_layout(phrases, total=4.0, spoken={0: 3.0, 1: 1.0}))
    check("ускорение: сработало", sped == 1, f"ускорено {sped}")
    check("ускорение: коэффициент в пределах потолка", phrases[0].atempo <= config.MAX_ATEMPO)
    check("ускорение: коэффициент 1.5 (needed=1.5)", approx(phrases[0].atempo, 1.5, 0.01))


def test_layout_below_threshold() -> None:
    """Превышение мелкое (ниже ATEMPO_MIN) — не трогаем, темп не скачет."""
    phrases = [
        Phrase(start=0.0, end=2.0, source="a", text="почти влезает"),
        Phrase(start=2.0, end=4.0, source="b", text="вторая"),
    ]
    _, _, sped, _, _ = asyncio.run(run_layout(phrases, total=4.0, spoken={0: 2.1, 1: 1.0}))
    check("порог: мелкое превышение не ускоряем", sped == 0, f"ускорено {sped}")
    check("порог: atempo остался 1.0", phrases[0].atempo == 1.0)


def test_layout_cap() -> None:
    """Нужно ускорение больше потолка — упираемся в MAX_ATEMPO, фраза наезжает."""
    phrases = [Phrase(start=0.0, end=1.0, source="a", text="очень длинная фраза")]
    _, _, sped, _, _ = asyncio.run(run_layout(phrases, total=1.0, spoken={0: 10.0}))
    check("потолок: ускорение ограничено", approx(phrases[0].atempo, config.MAX_ATEMPO, 0.01))
    check("потолок: фраза не выброшена", sped == 1)


def test_layout_empty_text() -> None:
    """Пустая строка не должна ломать укладку."""
    phrases = [
        Phrase(start=0.0, end=1.0, source="a", text=""),
        Phrase(start=1.0, end=2.0, source="b", text="есть"),
    ]
    dur, chars, _, _, _ = asyncio.run(run_layout(phrases, total=3.0, spoken={0: 1.0}))
    check("пустая фраза: пропущена", chars == len("есть"), str(chars))
    check("пустая фраза: длина дорожки цела", approx(dur, 3.0, 0.1), f"{dur:.2f}")


# ── 4. Текстовые форматы ────────────────────────────────────────────────────


def test_srt() -> None:
    phrases = [
        Phrase(start=0.0, end=1.5, source="a", text="Первая"),
        Phrase(start=3661.25, end=3663.0, source="b", text="Вторая"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = textout.write_srt(phrases, Path(tmp) / "s.srt")
        body = path.read_text(encoding="utf-8")
    check("srt: нумерация", body.startswith("1\n"))
    check("srt: таймкод от нуля", "00:00:00,000 --> 00:00:01,500" in body, body.splitlines()[1])
    check("srt: часы и миллисекунды", "01:01:01,250 --> 01:01:03,000" in body)


def test_paragraphs() -> None:
    phrases = [
        Phrase(start=0.0, end=1.0, source="", text="Первое."),
        Phrase(start=1.1, end=2.0, source="", text="Второе."),
        # пауза 3 с — новый абзац
        Phrase(start=5.0, end=6.0, source="", text="Третье."),
    ]
    paras = textout.build_paragraphs(phrases)
    check("абзацы: разрыв по паузе", len(paras) == 2, str(len(paras)))
    check("абзацы: первый склеен", paras[0] == "Первое. Второе.", paras[0])

    long_phrases = [
        Phrase(start=i * 0.5, end=i * 0.5 + 0.4, source="", text="слово" * 40)
        for i in range(6)
    ]
    paras = textout.build_paragraphs(long_phrases)
    check("абзацы: разрыв по длине", len(paras) > 1, str(len(paras)))


# ── 5. Смета ────────────────────────────────────────────────────────────────


def test_cost() -> None:
    # 5 минут: 300 с → 20 блоков по 15 с
    cost = speechkit.estimate_cost(300.0, 4000, 4200)
    expected_stt = 20 * config.STT_RUB_PER_15SEC
    check("смета: STT по блокам 15 с", approx(cost.stt, expected_stt, 0.001), f"{cost.stt:.4f}")
    check(
        "смета: Translate за символы",
        approx(cost.translate, 4000 / 1e6 * config.TRANSLATE_RUB_PER_MCHARS, 0.001),
    )
    check("смета: сумма сходится", approx(cost.total, cost.stt + cost.translate + cost.tts, 1e-9))

    # неполный блок округляется вверх
    check("смета: неполный блок вверх", speechkit.estimate_cost(1.0, 0, 0).stt == config.STT_RUB_PER_15SEC)

    # ориентир из ТЗ: 5 минут ≈ 12 ₽
    real = speechkit.estimate_cost(300.0, 4500, 4500)
    check("смета: 5 мин в районе 12 ₽", 8 <= real.total <= 18, f"{real.total:.2f} ₽")


# ── 6. Разбор ответа ИИ-редактора ───────────────────────────────────────────


def test_editor_parse() -> None:
    import editor

    good = "1⟩Первая строка\n2⟩Вторая строка\n3⟩Третья"
    check("редактор: разбор корректного ответа", editor._parse_reply(good, 3) is not None)

    short = "1⟩Первая\n2⟩Вторая"
    check("редактор: недобор строк отбит", editor._parse_reply(short, 3) is None)

    noisy = "Вот результат:\n1⟩Первая\n2⟩Вторая\n3⟩Третья\nГотово!"
    parsed = editor._parse_reply(noisy, 3)
    check("редактор: мусор вокруг строк отброшен", parsed == ["Первая", "Вторая", "Третья"])

    # глоссарий
    with tempfile.TemporaryDirectory() as tmp:
        gpath = Path(tmp) / "g.txt"
        gpath.write_text("# comment\nbull market = бычий рынок\nwhite paper|доклад\n", encoding="utf-8")
        terms = editor.load_glossary(gpath)
    check("глоссарий: два формата разделителя", terms == {"bull market": "бычий рынок", "white paper": "доклад"}, str(terms))

    hits = editor._relevant_terms({"bull market": "бычий рынок", "REIT": "фонд"}, ["we are in a bull market now"])
    check("глоссарий: в промт идут только встретившиеся термины", hits == ["bull market — бычий рынок"], str(hits))


# ── 7. Санитайзер имени файла ───────────────────────────────────────────────


def test_safe_name() -> None:
    import bot

    check("имя файла: слеши вырезаны", "/" not in bot.safe_name("a/b:c*d?"))
    check("имя файла: кириллица цела", bot.safe_name("Как жить") == "Как_жить")
    check("имя файла: пустое не пустое", bot.safe_name("???") == "translation")
    check("имя файла: длина ограничена", len(bot.safe_name("я" * 200)) <= 80)


# ── 8. Название ролика ──────────────────────────────────────────────────────


def test_nice_title() -> None:
    import media

    junk = [
        "a3f9c1b2d4e5f607",
        "5f3a9c1b2d4e.mp4",
        "1a2b3c4d5e6f7g8h9i0j",
        "",
        "https://cdn.example.com/x.mp4",
    ]
    for name in junk:
        check(f"мусорное имя: {name or 'пустое'}", media.is_junk_title(name))

    good = ["How to invest", "Разбор отчёта", "Meb Faber interview", "Bogleheads"]
    for name in good:
        check(f"нормальное имя: {name}", not media.is_junk_title(name))

    check(
        "nice_title берёт title",
        media.nice_title({"title": "Пассивные инвестиции"}, "https://x.com/a")
        == "Пассивные инвестиции",
    )
    check(
        "nice_title отбрасывает хеш",
        media.nice_title({"title": "a3f9c1b2d4e5f607"}, "https://x.com/a") == "",
    )
    check(
        "nice_title подхватывает track",
        media.nice_title({"title": "9f8e7d6c5b4a3210", "track": "Выпуск 12"}, "u")
        == "Выпуск 12",
    )
    check(
        "теги файла: артист и название",
        media.title_from_tags({"TITLE": "Эпизод 3", "artist": "Podcast"})
        == "Podcast — Эпизод 3",
    )
    check("теги файла: хеш отброшен", media.title_from_tags({"title": "deadbeef1234"}) == "")
    check(
        "fallback: домен в имени",
        "example.com" in media.fallback_title("https://www.example.com/a/b.mp4"),
    )


# ── 9. Перебор клиентов yt-dlp ──────────────────────────────────────────────


def test_ytdlp_retry() -> None:
    import media

    retryable = [
        "ERROR: [youtube] abc: Sign in to confirm you're not a bot. Use --cookies",
        "ERROR: [youtube] abc: The page needs to be reloaded.",
        "ERROR: [youtube] abc: Requested format is not available.",
        "ERROR: unable to download webpage: HTTP Error 429: Too Many Requests",
        "ERROR: [youtube] abc: Failed to extract any player response",
    ]
    for text in retryable:
        check(f"повтор нужен: {text[:40]}", media.is_retryable_error(text))

    final = [
        "ERROR: [youtube] abc: Video unavailable. This video is private",
        "ERROR: [generic] Unsupported URL: file:///etc/passwd",
        "ERROR: unable to open for writing: No space left on device",
    ]
    for text in final:
        check(f"повтор бесполезен: {text[:40]}", not media.is_retryable_error(text))

    check("наборов клиентов больше одного", len(media.YTDLP_CLIENTS) >= 2)
    check(
        "первый набор — рабочий на этом сервере",
        media.YTDLP_CLIENTS[0] == "default,android_vr",
    )
    check(
        "клиент подставляется в extractor-args",
        "youtube:player_client=android_vr" in media._ytdlp_base("android_vr"),
    )


if __name__ == "__main__":
    test_phrases()
    test_batching()
    test_layout_fits()
    test_layout_speedup()
    test_layout_below_threshold()
    test_layout_cap()
    test_layout_empty_text()
    test_srt()
    test_paragraphs()
    test_cost()
    test_editor_parse()
    test_safe_name()
    test_nice_title()
    test_ytdlp_retry()

    print()
    if FAILED:
        print(f"Провалено проверок: {len(FAILED)}")
        for name in FAILED:
            print("  -", name)
        sys.exit(1)
    print("Все проверки пройдены.")
