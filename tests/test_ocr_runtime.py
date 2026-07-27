from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import marriage_ocr.ocr as ocr_module
from marriage_ocr.ocr import (
    _build_paddle_constructor_kwargs,
    _build_paddle_install_guidance,
    _google_vision_annotation_to_result,
    _normalize_paddle_output,
    _run_paddle_inference,
    RecordCropPaths,
)
from marriage_ocr.layout import Box
from marriage_ocr.models import OcrLine, OcrResult


def test_paddle_guidance_mentions_unsupported_python() -> None:
    message = _build_paddle_install_guidance(python_version=(3, 14), system="Linux", machine="x86_64")

    assert "Current Python is 3.14" in message
    assert "Python 3.9-3.13" in message


def test_paddle_guidance_mentions_apple_silicon_constraints() -> None:
    message = _build_paddle_install_guidance(python_version=(3, 11), system="Darwin", machine="arm64")

    assert "macOS arm64" in message
    assert "macOS CPU installation path" in message


def test_paddle_constructor_kwargs_map_angle_flag_for_newer_api() -> None:
    class FakePaddleOCR:
        def __init__(self, lang=None, use_textline_orientation=None, text_recognition_batch_size=None):
            pass

    kwargs = _build_paddle_constructor_kwargs(
        paddle_ocr_class=FakePaddleOCR,
        lang="en",
        use_angle_cls=True,
        show_log=False,
        extra_config={"text_recognition_batch_size": 4, "show_log": True, "unknown_flag": 1},
    )

    assert kwargs == {
        "lang": "en",
        "use_textline_orientation": True,
        "text_recognition_batch_size": 4,
    }


def test_run_paddle_inference_prefers_predict_for_newer_api() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def predict(self, input, *, use_textline_orientation=None):
            self.calls.append((input, {"use_textline_orientation": use_textline_orientation}))
            return [{"rec_texts": ["abc"], "rec_scores": [0.9], "rec_boxes": [[1, 2, 3, 4]]}]

    engine = FakeEngine()
    result = _run_paddle_inference(engine, "sample.jpg", use_angle_cls=True)

    assert engine.calls == [("sample.jpg", {"use_textline_orientation": True})]
    assert result[0]["rec_texts"] == ["abc"]


def test_normalize_paddle_output_supports_modern_result_shape() -> None:
    normalized = _normalize_paddle_output(
        [
            {
                "rec_texts": ["BIL", "NAMA"],
                "rec_scores": [0.81, 0.92],
                "rec_boxes": [[1, 2, 31, 20], [40, 3, 88, 25]],
            }
        ]
    )

    assert normalized == [
        {"text": "BIL", "confidence": 0.81, "bbox": [1.0, 2.0, 31.0, 20.0]},
        {"text": "NAMA", "confidence": 0.92, "bbox": [40.0, 3.0, 88.0, 25.0]},
    ]


def test_normalize_paddle_output_handles_empty_modern_result() -> None:
    normalized = _normalize_paddle_output(
        [
            {
                "rec_texts": [],
                "rec_scores": [],
                "rec_boxes": [],
            }
        ]
    )

    assert normalized == []


def test_google_vision_annotation_to_result_builds_lines_and_bbox() -> None:
    word_1 = SimpleNamespace(
        symbols=[SimpleNamespace(text="ALI")],
        confidence=0.8,
        bounding_box=SimpleNamespace(
            vertices=[
                SimpleNamespace(x=10, y=20),
                SimpleNamespace(x=30, y=20),
                SimpleNamespace(x=30, y=40),
                SimpleNamespace(x=10, y=40),
            ]
        ),
    )
    word_2 = SimpleNamespace(
        symbols=[SimpleNamespace(text="BIN"), SimpleNamespace(text=" " ), SimpleNamespace(text="AHMAD")],
        confidence=0.6,
        bounding_box=SimpleNamespace(
            vertices=[
                SimpleNamespace(x=35, y=22),
                SimpleNamespace(x=80, y=22),
                SimpleNamespace(x=80, y=42),
                SimpleNamespace(x=35, y=42),
            ]
        ),
    )
    annotation = SimpleNamespace(
        text="ALI BIN AHMAD",
        pages=[
            SimpleNamespace(
                blocks=[
                    SimpleNamespace(
                        paragraphs=[
                            SimpleNamespace(
                                words=[word_1, word_2],
                                confidence=0.0,
                            )
                        ]
                    )
                ]
            )
        ],
    )

    result = _google_vision_annotation_to_result(annotation)

    assert result.text == "ALI BIN AHMAD"
    assert result.average_confidence == 0.7
    assert len(result.lines) == 1
    assert result.lines[0].text == "ALI BIN AHMAD"
    assert result.lines[0].bbox == [10.0, 20.0, 80.0, 42.0]


def test_build_ocr_engine_supports_google_vision_aliases() -> None:
    sentinel = object()
    with patch.object(ocr_module, "GoogleVisionOcrEngine", autospec=True) as google_cls:
        google_cls.return_value = sentinel

        engine = ocr_module.build_ocr_engine(
            {
                "engine": "vision",
                "google_vision": {"language_hints": ["ms", "en"]},
            }
        )

    google_cls.assert_called_once_with({"language_hints": ["ms", "en"]})
    assert engine is sentinel


def test_run_ocr_on_page_layout_uses_crop_fallback_for_weak_assignment(tmp_path: Path) -> None:
    page_path = tmp_path / "page.jpg"
    page_path.write_text("page", encoding="utf-8")

    record_dir = tmp_path / "record_001"
    record_dir.mkdir()
    crop_path = record_dir / "bil.jpg"
    crop_path.write_text("crop", encoding="utf-8")

    layout = SimpleNamespace(records=[SimpleNamespace(cells={"bil": Box.from_bounds(10, 10, 30, 30)})])
    records = [
        RecordCropPaths(
            record_index=1,
            record_dir=record_dir,
            full_record_path=record_dir / "full_record.jpg",
            cell_paths={"bil": crop_path},
        )
    ]

    class FakeEngine:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def read_image(self, image_path: str | Path) -> OcrResult:
            path = Path(image_path)
            self.calls.append(path.name)

            if path.name == "page.jpg":
                return OcrResult(text="", lines=[], average_confidence=0.12)
            if path.name == "bil.jpg":
                return OcrResult(
                    text="12/94",
                    lines=[OcrLine(text="12/94", confidence=0.92, bbox=[1.0, 1.0, 20.0, 20.0])],
                    average_confidence=0.92,
                )
            raise AssertionError(f"Unexpected OCR path: {path}")

    engine = FakeEngine()
    outputs = ocr_module.run_ocr_on_page_layout(
        page_path,
        layout,
        records,
        engine,
        save_raw_json=False,
    )

    assert engine.calls == ["page.jpg", "bil.jpg"]
    assert outputs[0].cell_results["bil"].text == "12/94"
