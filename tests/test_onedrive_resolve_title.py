from __future__ import annotations

from types import SimpleNamespace

import pytest

from marriage_ocr import onedrive_ingest


def _response(url: str, headers: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        url=url, headers=headers or {}, raise_for_status=lambda: None, close=lambda: None
    )


@pytest.mark.parametrize(
    ("final_url", "headers", "expected"),
    [
        (
            "https://onedrive.live.com/?id=A%21s1&cid=A&spopath=%2Fpersonal%2FA%2FDocuments%2FSCAN%2FBUKU%202014%20%283%29&ithint=folder",
            {},
            "BUKU 2014 (3)",
        ),
        ("https://onedrive.live.com/?id=%2Fpersonal%2FA%2FDocuments%2FNikah", {}, "Nikah"),
        (
            "https://x.sharepoint.com/download.aspx?a=1",
            {"Content-Disposition": 'attachment; filename="Gombak.zip"'},
            "Gombak",
        ),
        ("https://onedrive.live.com/?cid=A", {}, None),
        ("https://login.live.com/oauth?id=%2Fa%2Fb", {}, None),
    ],
)
def test_resolve_share_title(monkeypatch, final_url, headers, expected) -> None:
    monkeypatch.setattr(onedrive_ingest.requests, "get", lambda *a, **k: _response(final_url, headers))
    assert onedrive_ingest.resolve_share_title("https://1drv.ms/f/x") == expected
