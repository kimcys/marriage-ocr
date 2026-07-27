from pathlib import Path
from types import SimpleNamespace

from google.api_core.exceptions import ServiceUnavailable

from marriage_ocr.typed.models import RenderedPage
from marriage_ocr.typed.vision import TypedVisionClient, annotation_to_page_result


def _vertex(x: int, y: int) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


def _word(text: str, confidence: float, box: tuple[int, int, int, int]) -> SimpleNamespace:
    symbols = [SimpleNamespace(text=character) for character in text]
    x1, y1, x2, y2 = box
    return SimpleNamespace(
        symbols=symbols,
        confidence=confidence,
        bounding_box=SimpleNamespace(
            vertices=[_vertex(x1, y1), _vertex(x2, y1), _vertex(x2, y2), _vertex(x1, y2)]
        ),
    )


def test_annotation_to_page_result_normalises_word_coordinates() -> None:
    annotation = SimpleNamespace(
        text="HENDON",
        pages=[
            SimpleNamespace(
                width=1000,
                height=2000,
                blocks=[SimpleNamespace(paragraphs=[SimpleNamespace(words=[_word("HENDON", 0.98, (100, 200, 300, 260))])])],
            )
        ],
    )

    result = annotation_to_page_result(annotation, source_file="record.pdf", page_number=1)

    word = result.words[0]
    assert word.text == "HENDON"
    assert word.confidence == 0.98
    assert (word.x1, word.y1, word.x2, word.y2) == (0.1, 0.1, 0.3, 0.13)


def test_client_retries_transient_batch_error(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    page = RenderedPage(tmp_path / "record.pdf", "record.pdf", 1, image_path, 100, 100)
    calls = {"count": 0}

    class FakeClient:
        def batch_annotate_images(self, *, requests):
            calls["count"] += 1
            if calls["count"] < 3:
                raise ServiceUnavailable("temporary")
            annotation = SimpleNamespace(text="", pages=[])
            return SimpleNamespace(
                responses=[SimpleNamespace(error=SimpleNamespace(message=""), full_text_annotation=annotation)]
            )

    monkeypatch.setattr("time.sleep", lambda seconds: None)
    client = TypedVisionClient(client=FakeClient(), vision_module=SimpleNamespace(), api_attempts=3)
    client._build_request = lambda path: object()

    results = client.annotate_pages([page])

    assert len(results) == 1
    assert calls["count"] == 3

