from __future__ import annotations

import base64
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

import marriage_ocr.onedrive_ingest as onedrive_ingest
from marriage_ocr.onedrive_ingest import (
    _add_download_param,
    _filename_from_response,
    download_anonymous_share,
    encode_share_url,
)


def test_encode_share_url_uses_documented_u_bang_scheme_and_round_trips():
    url = "https://contoso-my.sharepoint.com/:f:/g/personal/user_contoso_com/AbCdEf?e=XyZ123"

    encoded = encode_share_url(url)

    assert encoded.startswith("u!")
    body = encoded[2:]
    # Unpadded base64url: no '+', '/', or '=' characters allowed.
    assert "+" not in body
    assert "/" not in body
    assert "=" not in body

    # Reverse Microsoft's documented transform and confirm it recovers the
    # exact original URL -- the real correctness property that matters here.
    restored_b64 = body.replace("-", "+").replace("_", "/")
    padding = "=" * (-len(restored_b64) % 4)
    decoded = base64.b64decode(restored_b64 + padding).decode("utf-8")
    assert decoded == url


def test_encode_share_url_strips_surrounding_whitespace():
    encoded_with_whitespace = encode_share_url("  https://example.com/share  ")
    encoded_clean = encode_share_url("https://example.com/share")
    assert encoded_with_whitespace == encoded_clean


def test_add_download_param_appends_to_existing_query():
    url = "https://contoso-my.sharepoint.com/:w:/g/personal/user_contoso_com/AbCdEf?e=XyZ123"
    assert _add_download_param(url) == url + "&download=1"


def test_add_download_param_adds_query_when_none_present():
    assert _add_download_param("https://1drv.ms/x/s!abc") == "https://1drv.ms/x/s!abc?download=1"


class _FakeResponse:
    def __init__(self, url, *, headers=None, content=b"", status_code=200):
        self.url = url
        self.headers = headers or {}
        self.status_code = status_code
        self._content = content

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self._content


def test_filename_from_response_prefers_content_disposition():
    response = _FakeResponse(
        "https://contoso-my.sharepoint.com/download?download=1",
        headers={"Content-Disposition": 'attachment; filename="Marriage Cert.pdf"'},
    )
    assert _filename_from_response(response, fallback="download") == "Marriage Cert.pdf"


def test_filename_from_response_falls_back_to_url_path():
    response = _FakeResponse("https://contoso-my.sharepoint.com/files/scan.jpg")
    assert _filename_from_response(response, fallback="download") == "scan.jpg"


def test_download_anonymous_share_saves_single_file(monkeypatch, tmp_path: Path):
    file_bytes = b"%PDF-1.4 fake content"
    resolve_response = _FakeResponse("https://contoso-my.sharepoint.com/:b:/g/personal/x/AbCdEf?e=1")
    download_response = _FakeResponse(
        "https://contoso-my.sharepoint.com/:b:/g/personal/x/AbCdEf?e=1&download=1",
        headers={
            "Content-Disposition": 'attachment; filename="record.pdf"',
            "Content-Type": "application/pdf",
        },
        content=file_bytes,
    )
    calls = iter([resolve_response, download_response])
    monkeypatch.setattr(onedrive_ingest.requests, "get", lambda *a, **kw: next(calls))

    downloaded = download_anonymous_share("https://1drv.ms/b/s!AbCdEf", tmp_path)

    assert downloaded == [tmp_path / "record.pdf"]
    assert (tmp_path / "record.pdf").read_bytes() == file_bytes


def test_download_anonymous_share_extracts_zip_folder(monkeypatch, tmp_path: Path):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("page1.jpg", b"jpg-bytes")
        archive.writestr("sub/page2.jpg", b"jpg-bytes-2")
    zip_bytes = buffer.getvalue()

    resolve_response = _FakeResponse("https://contoso-my.sharepoint.com/:f:/g/personal/x/AbCdEf?e=1")
    download_response = _FakeResponse(
        "https://contoso-my.sharepoint.com/:f:/g/personal/x/AbCdEf?e=1&download=1",
        headers={
            "Content-Disposition": 'attachment; filename="folder.zip"',
            "Content-Type": "application/zip",
        },
        content=zip_bytes,
    )
    calls = iter([resolve_response, download_response])
    monkeypatch.setattr(onedrive_ingest.requests, "get", lambda *a, **kw: next(calls))

    downloaded = download_anonymous_share("https://1drv.ms/f/s!AbCdEf", tmp_path)

    assert sorted(p.relative_to(tmp_path).as_posix() for p in downloaded) == [
        "page1.jpg",
        "sub/page2.jpg",
    ]
    assert (tmp_path / "page1.jpg").read_bytes() == b"jpg-bytes"
    assert (tmp_path / "sub" / "page2.jpg").read_bytes() == b"jpg-bytes-2"


def test_download_anonymous_share_raises_when_link_requires_signin(monkeypatch, tmp_path: Path):
    resolve_response = _FakeResponse("https://contoso-my.sharepoint.com/:b:/g/personal/x/AbCdEf?e=1")
    signin_response = _FakeResponse(
        "https://login.microsoftonline.com/common/oauth2/authorize?...",
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=b"<html>sign in</html>",
    )
    calls = iter([resolve_response, signin_response])
    monkeypatch.setattr(onedrive_ingest.requests, "get", lambda *a, **kw: next(calls))
    browser_fallback_called = False

    def fake_browser_fallback(*a, **kw):
        nonlocal browser_fallback_called
        browser_fallback_called = True
        return []

    monkeypatch.setattr(onedrive_ingest, "_download_via_browser", fake_browser_fallback)

    with pytest.raises(RuntimeError, match="requires signing in"):
        download_anonymous_share("https://1drv.ms/b/s!AbCdEf", tmp_path)
    # A genuine sign-in-host redirect can't be helped by a browser either --
    # the fallback must not even be attempted for this case.
    assert browser_fallback_called is False


def test_download_anonymous_share_falls_back_to_browser_for_html_non_signin_response(
    monkeypatch, tmp_path: Path
):
    """The ambiguous shape found empirically: HTML body, but the final host
    is OneDrive itself, not a Microsoft sign-in host -- e.g. a folder link
    migrated to SharePoint Online that's genuinely browsable anonymously but
    whose plain GET only returns the web-app shell."""
    resolve_response = _FakeResponse("https://onedrive.live.com/?id=abc")
    html_response = _FakeResponse(
        "https://onedrive.live.com/?id=abc&download=1",
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=b"<html>onedrive web app shell</html>",
    )
    calls = iter([resolve_response, html_response])
    monkeypatch.setattr(onedrive_ingest.requests, "get", lambda *a, **kw: next(calls))

    fallback_calls: list[tuple[str, Path]] = []

    def fake_browser_fallback(share_url, dest_dir):
        fallback_calls.append((share_url, dest_dir))
        return [dest_dir / "recovered.pdf"]

    monkeypatch.setattr(onedrive_ingest, "_download_via_browser", fake_browser_fallback)

    downloaded = download_anonymous_share("https://1drv.ms/f/s!AbCdEf", tmp_path)

    assert downloaded == [tmp_path / "recovered.pdf"]
    assert fallback_calls == [("https://1drv.ms/f/s!AbCdEf", tmp_path)]
