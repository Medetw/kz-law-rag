from kzlaw.ingest.parse import MAX_CHARS, Article, chunk_article, parse_articles

from .conftest import TAX_KZ_HTML, TAX_RU_HTML


def test_articles_found_with_titles_paths_and_anchors() -> None:
    arts = {a.number: a for a in parse_articles(TAX_RU_HTML, "tax2026", "ru")}
    assert set(arts) == {"503", "504", "276"}
    a = arts["503"]
    assert a.title == "Ставки налога на добавленную стоимость"
    assert a.anchor == "z8609"
    assert a.path == [
        "ОСОБЕННАЯ ЧАСТЬ",
        "РАЗДЕЛ 7. НАЛОГ НА ДОБАВЛЕННУЮ СТОИМОСТЬ",
        "Глава 51. ПОРЯДОК ИСЧИСЛЕНИЯ И УПЛАТЫ НАЛОГА",
    ]
    assert arts["276"].path[-1] == "Глава 52. ГРУППЫ"


def test_editorial_notes_are_dropped() -> None:
    text = " ".join(a.text for a in parse_articles(TAX_RU_HTML, "tax2026", "ru"))
    assert "Сноска" not in text
    assert "ИЗПИ" not in text
    assert "16 процентов" in text


def test_tables_are_flattened_into_rows() -> None:
    a = {a.number: a for a in parse_articles(TAX_RU_HTML, "tax2026", "ru")}["276"]
    assert "I | Здания, сооружения" in a.text


def test_kazakh_article_headers_in_h3() -> None:
    (a,) = parse_articles(TAX_KZ_HTML, "tax2026", "kz")
    assert a.number == "503"
    assert a.title == "Қосылған құн салығының мөлшерлемелері"
    assert a.anchor == "z8611"
    assert "16 пайызды" in a.text


def test_chunk_carries_context_header_points_and_deep_link() -> None:
    a = {a.number: a for a in parse_articles(TAX_RU_HTML, "tax2026", "ru")}["503"]
    (c,) = chunk_article(a)
    assert c.id == "tax2026:503:ru:0"
    assert c.cite_id == "tax2026:503"
    assert c.points == ("1", "2")
    assert c.url == "https://adilet.zan.kz/rus/docs/K2500000214#z8609"
    assert c.full_text.startswith("Налоговый кодекс РК (редакция с 2026 года). Статья 503.")


def test_long_articles_split_on_point_boundaries_under_limit() -> None:
    paras = [f"{i}. " + "слово " * 120 for i in range(1, 9)]  # ~730 chars each
    a = Article(code="labor", lang="ru", number="88", title="Отпуск", anchor=None, path=[], paragraphs=paras)
    chunks = chunk_article(a)
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHARS for c in chunks)
    assert [p for c in chunks for p in c.points] == [str(i) for i in range(1, 9)]
    assert [c.part for c in chunks] == list(range(len(chunks)))


def test_giant_paragraph_is_split_too() -> None:
    a = Article(
        code="labor",
        lang="ru",
        number="1",
        title="Х",
        anchor=None,
        path=[],
        paragraphs=["Предложение номер один. " * 400],
    )
    assert all(len(c.text) <= MAX_CHARS for c in chunk_article(a))
