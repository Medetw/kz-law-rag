from datetime import date

import pytest

from kzlaw.agent import citations
from kzlaw.corpus import codes_in_force
from kzlaw.embeddings import HashEmbedder
from kzlaw.search.bm25 import BM25
from kzlaw.search.expand import expand, strip_years
from kzlaw.search.hybrid import rrf
from kzlaw.search.text import is_kazakh, tokenize


def test_russian_words_are_stemmed_and_stopwords_dropped() -> None:
    assert tokenize("Ставки налога на добавленную стоимость") == tokenize("ставка налогов добавленной стоимости")
    assert "на" not in tokenize("налог на добавленную стоимость")


def test_numbers_survive_tokenisation() -> None:
    assert {"503", "16"} <= set(tokenize("Статья 503: 16 процентов"))


def test_kazakh_detection_and_prefix_stems() -> None:
    assert is_kazakh("Қосылған құн салығы")
    assert not is_kazakh("налог на добавленную стоимость")
    # one stem with ғ/қ alternation -> same 6-letter prefix; different case endings may still differ
    assert tokenize("салығы")[0] == tokenize("салығының")[0] == "салықы"


@pytest.mark.parametrize(
    ("query", "must_contain"),
    [
        ("Какая ставка НДС?", "налог на добавленную стоимость"),
        ("сколько длится декрет", "отпуск по беременности и родам"),
        ("ҚҚС мөлшерлемесі", "қосылған құн салығы"),
        ("VAT rate", "налог на добавленную стоимость"),
    ],
)
def test_query_expansion(query: str, must_contain: str) -> None:
    assert must_contain in expand(query)


def test_expansion_leaves_plain_queries_alone() -> None:
    assert expand("ставка налога на добавленную стоимость") == "ставка налога на добавленную стоимость"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Какая ставка НДС действовала в 2025 году?", "Какая ставка НДС действовала ?"),
        ("2026 жылы ҚҚС мөлшерлемесі", "ҚҚС мөлшерлемесі"),
        ("VAT rate in 2026", "VAT rate"),
        ("статья 503", "статья 503"),
    ],
)
def test_years_are_removed_from_queries(query: str, expected: str) -> None:
    assert strip_years(query) == expected


def test_bm25_prefers_rare_matching_terms() -> None:
    docs = [
        "налог на добавленную стоимость налог на добавленную стоимость",
        "налоговый период по налогу на добавленную стоимость календарный квартал",
        "трудовой отпуск двадцать четыре дня",
    ]
    top = BM25(docs).top("налоговый период", k=3)
    assert top[0][0] == 1


def test_bm25_mask_and_empty_index() -> None:
    import numpy as np

    bm = BM25(["отпуск", "отпуск работника"])
    assert [i for i, _ in bm.top("отпуск", 5, np.array([False, True]))] == [1]
    assert BM25([]).top("что угодно", 5) == []


def test_rrf_rewards_agreement_between_rankings() -> None:
    fused = rrf([["a", "b", "c"], ["b", "c", "a"]])
    assert max(fused, key=fused.__getitem__) == "b"
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_hash_embedder_is_normalised_and_lexical() -> None:
    e = HashEmbedder(64)
    a, b, c = e.embed_documents(["ставка налога", "ставки налогов", "трудовой отпуск"])
    assert float(a @ a) == pytest.approx(1.0)
    assert float(a @ b) > float(a @ c)


def test_codes_in_force_switch_on_1_january_2026() -> None:
    assert codes_in_force(date(2025, 12, 31)) == ["tax2017", "labor"]
    assert codes_in_force(date(2026, 1, 1)) == ["tax2026", "labor"]


def test_citation_extraction_and_stripping() -> None:
    text = "Ставка 16% [tax2026:503]. Раньше 12% [tax2017:422, tax2026:999]. Отпуск [labor:88]; ещё [tax2026:999]."
    assert citations.extract(text) == ["tax2026:503", "tax2017:422", "tax2026:999", "labor:88"]
    cleaned = citations.strip(text, {"tax2026:999"})
    assert "999" not in cleaned
    assert "[tax2017:422]" in cleaned
    assert cleaned.endswith("ещё.")
