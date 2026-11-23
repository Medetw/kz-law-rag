import pytest

from kzlaw.agent import quotes

ART_56 = (
    "1. Работник вправе по своей инициативе расторгнуть трудовой договор, уведомив об этом работодателя "
    "не менее чем за один месяц. 3. Работодатель обязан расторгнуть трудовой договор по истечении семи "
    "рабочих дней со дня уведомления — в случае нарушения работодателем условий договора."
)


def test_short_quotes_are_terms_not_quotes() -> None:
    assert quotes.extract("Закон «О гражданстве» и «НДС»") == []
    assert quotes.extract("Срок — «не менее чем за один месяц»") == ["не менее чем за один месяц"]


def test_exact_quote_passes() -> None:
    res = quotes.check("Нужно уведомить «не менее чем за один месяц» [labor:56].", [ART_56])
    assert res.verified == ["не менее чем за один месяц"]
    assert res.invalid == []


def test_changed_legal_wording_fails() -> None:
    res = quotes.check("Договор расторгается «в течение семи рабочих дней со дня уведомления».", [ART_56])
    assert res.invalid == ["в течение семи рабочих дней со дня уведомления"]


@pytest.mark.parametrize(
    "quote",
    [
        "«По истечении семи рабочих дней со дня уведомления»",  # capital letter
        "«по истечении семи  рабочих\nдней со дня уведомления.»",  # whitespace, trailing dot
        "“по истечении семи рабочих дней со дня уведомления - в случае нарушения”",  # other quotes, hyphen for dash
    ],
)
def test_typography_is_forgiven(quote: str) -> None:
    assert quotes.check(f"Так: {quote}", [ART_56]).invalid == []


def test_yo_and_e_are_the_same_letter() -> None:
    assert (
        quotes.check(
            "«ставка налога составляет 16 процентов, ещё раз»", ["Ставка налога составляет 16 процентов, еще раз"]
        ).invalid
        == []
    )


def test_ellipsis_parts_must_appear_in_order() -> None:
    ok = "«Работник вправе … уведомив об этом работодателя»"
    wrong_order = "«уведомив об этом работодателя … Работник вправе по своей инициативе»"
    assert quotes.check(ok, [ART_56]).invalid == []
    assert quotes.check(wrong_order, [ART_56]).invalid != []


def test_quote_may_come_from_any_source() -> None:
    res = quotes.check(
        "«налоговым периодом является календарный квартал»",
        [ART_56, "Налоговым периодом является календарный квартал."],
    )
    assert res.invalid == []


def test_unquote_keeps_text_but_drops_the_marks() -> None:
    text = "Срок «в течение семи рабочих дней со дня уведомления» [labor:56]."
    assert quotes.unquote(text, ["в течение семи рабочих дней со дня уведомления"]) == (
        "Срок в течение семи рабочих дней со дня уведомления [labor:56]."
    )


def test_spaces_left_by_removed_links_are_forgiven() -> None:
    source = "за исключением соответствующих подпунктам 1) , 2) и 3) пункта 2 ( далее – организации)"
    assert (
        quotes.check("«соответствующих подпунктам 1), 2) и 3) пункта 2 (далее – организации)»", [source]).invalid == []
    )
