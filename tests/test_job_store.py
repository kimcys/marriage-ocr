import json
from pathlib import Path

from marriage_ocr.job_store import OcrJob, create_job, list_jobs, load_job, save_job


def test_create_job_creates_directories_and_metadata(tmp_path: Path) -> None:
    job = create_job(tmp_path)

    job_dir = tmp_path / job.job_id
    metadata_path = job_dir / "metadata.json"

    assert job.status == "CREATED"
    assert metadata_path.exists()
    assert Path(job.input_dir).is_dir()
    assert Path(job.output_dir).is_dir()
    assert Path(job.debug_dir).is_dir()

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["job_id"] == job.job_id
    assert payload["status"] == "CREATED"


def test_create_job_can_skip_debug_dir(tmp_path: Path) -> None:
    job = create_job(tmp_path, create_debug_dir=False)

    assert not Path(job.debug_dir).exists()


def test_save_and_load_job_round_trip(tmp_path: Path) -> None:
    job = OcrJob(
        job_id="20260525_120000_abc12345",
        status="RUNNING",
        created_at="2020-01-01T00:00:00",
        updated_at="2020-01-01T00:00:00",
        input_dir=str(tmp_path / "20260525_120000_abc12345" / "input"),
        output_dir=str(tmp_path / "20260525_120000_abc12345" / "output"),
        debug_dir=str(tmp_path / "20260525_120000_abc12345" / "debug"),
        message="Processing page 1",
        total_pages=3,
        total_detected_records=12,
        total_parsed_records=4,
    )

    save_job(job, root=tmp_path)
    loaded = load_job(job.job_id, root=tmp_path)

    assert loaded.job_id == job.job_id
    assert loaded.status == "RUNNING"
    assert loaded.created_at == "2020-01-01T00:00:00"
    assert loaded.updated_at >= "2020-01-01T00:00:00"
    assert loaded.message == "Processing page 1"
    assert loaded.total_pages == 3
    assert loaded.total_detected_records == 12
    assert loaded.total_parsed_records == 4


def test_list_jobs_returns_newest_first(tmp_path: Path) -> None:
    older = OcrJob(
        job_id="20260525_110000_old00001",
        status="DONE",
        created_at="2026-05-25T11:00:00",
        updated_at="2026-05-25T11:00:00",
        input_dir=str(tmp_path / "20260525_110000_old00001" / "input"),
        output_dir=str(tmp_path / "20260525_110000_old00001" / "output"),
        debug_dir=str(tmp_path / "20260525_110000_old00001" / "debug"),
    )
    newer = OcrJob(
        job_id="20260525_120000_new00002",
        status="CREATED",
        created_at="2026-05-25T12:00:00",
        updated_at="2026-05-25T12:00:00",
        input_dir=str(tmp_path / "20260525_120000_new00002" / "input"),
        output_dir=str(tmp_path / "20260525_120000_new00002" / "output"),
        debug_dir=str(tmp_path / "20260525_120000_new00002" / "debug"),
    )

    save_job(older, root=tmp_path)
    save_job(newer, root=tmp_path)

    jobs = list_jobs(tmp_path)

    assert [job.job_id for job in jobs] == [newer.job_id, older.job_id]


def test_list_jobs_returns_empty_when_root_missing(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    assert list_jobs(missing_root) == []
