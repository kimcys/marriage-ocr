from __future__ import annotations

import base64

from marriage_ocr.onedrive_ingest import encode_share_url


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
