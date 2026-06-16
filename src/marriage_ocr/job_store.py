from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4


DEFAULT_JOBS_ROOT = Path("data/jobs")


@dataclass
class OcrJob:
    job_id: str
    status: str
    created_at: str
    updated_at: str
    input_dir: str
    output_dir: str
    debug_dir: str
    message: str = ""
    total_pages: int = 0
    total_detected_records: int = 0
    total_parsed_records: int = 0


def create_job(root: Path = DEFAULT_JOBS_ROOT, *, create_debug_dir: bool = True) -> OcrJob:
    now = _timestamp_now()
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    job_dir = root / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    debug_dir = job_dir / "debug"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if create_debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    job = OcrJob(
        job_id=job_id,
        status="CREATED",
        created_at=now,
        updated_at=now,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        debug_dir=str(debug_dir),
    )
    save_job(job, root=root)
    return job


def save_job(job: OcrJob, root: Path = DEFAULT_JOBS_ROOT) -> None:
    job.updated_at = _timestamp_now()
    metadata_path = _metadata_path(job.job_id, root)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(asdict(job), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_job(job_id: str, root: Path = DEFAULT_JOBS_ROOT) -> OcrJob:
    metadata_path = _metadata_path(job_id, root)
    return OcrJob(**json.loads(metadata_path.read_text(encoding="utf-8")))


def list_jobs(root: Path = DEFAULT_JOBS_ROOT) -> list[OcrJob]:
    if not root.exists():
        return []

    jobs = [
        OcrJob(**json.loads(metadata_path.read_text(encoding="utf-8")))
        for metadata_path in root.glob("*/metadata.json")
    ]
    return sorted(jobs, key=lambda job: (job.created_at, job.job_id), reverse=True)


def _metadata_path(job_id: str, root: Path) -> Path:
    return root / job_id / "metadata.json"


def _timestamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")
