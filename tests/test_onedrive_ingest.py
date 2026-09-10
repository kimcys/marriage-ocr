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


def test_download_anonymous_share_zip_skips_already_extracted_files(monkeypatch, tmp_path: Path):
    """Simulates retrying a submission whose destination dir already has a
    file from a prior (interrupted) attempt -- marriage-be reuses the same
    dest_dir across a retry (see onedrive/service.py::run_onedrive_fetch),
    so re-extracting a file that's already there byte-for-byte correct is
    wasted work, and re-writing it is exactly what a truncated-on-crash file
    would need to avoid becoming permanently mistaken for "done"."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("page1.jpg", b"jpg-bytes")
        archive.writestr("page2.jpg", b"jpg-bytes-2")
    zip_bytes = buffer.getvalue()

    # page1.jpg already exists locally, correct size -- must be left alone.
    (tmp_path / "page1.jpg").write_bytes(b"jpg-bytes")
    already_extracted_mtime = (tmp_path / "page1.jpg").stat().st_mtime_ns

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

    assert sorted(p.name for p in downloaded) == ["page1.jpg", "page2.jpg"]
    assert (tmp_path / "page1.jpg").stat().st_mtime_ns == already_extracted_mtime
    assert (tmp_path / "page2.jpg").read_bytes() == b"jpg-bytes-2"
    assert not (tmp_path / "page2.jpg.part").exists()


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


# --- Nested-folder / virtualized-listing browser fallback -----------------
#
# Real client share links turned out to be OneDrive folders that are
# themselves full of sub-folders (category -> district -> year -> files, up
# to 3-4 levels deep) rather than a flat list of files, and some individual
# folders hold hundreds or thousands of items. OneDrive's web UI virtualizes
# that listing (only ~30-60 rows exist in the DOM at once; scrolling its own
# container lazily grows how much is loaded) rather than rendering
# everything up front. These tests exercise that against lightweight fakes
# standing in for Playwright's page/locator objects -- no real browser or
# network involved.


class _FakeIcon:
    def __init__(self, aria_label):
        self._aria_label = aria_label

    def get_attribute(self, name):
        return self._aria_label if name == "aria-label" else None


class _FakeLocatorList:
    def __init__(self, items):
        self._items = items

    def count(self):
        return len(self._items)

    def nth(self, i):
        return self._items[i]

    @property
    def first(self):
        return self._items[0]


class _FakeTextCell:
    def __init__(self, text):
        self._text = text

    def count(self):
        return 1

    def inner_text(self):
        return self._text


class _FakeIconCell:
    def __init__(self, icons):
        self._icons = icons

    def locator(self, selector):
        assert selector == "i[role='img']"
        return _FakeLocatorList(self._icons)


class _Node:
    def __init__(self, name, is_folder, children=None):
        self.name = name
        self.is_folder = is_folder
        self.children = children or []


class _FakeMoreButton:
    def __init__(self, page, node):
        self._page = page
        self._node = node

    def click(self, timeout=None):
        self._page.download_attempts.append(self._node.name)
        self._page._menu_target = self._node


class _FakeMenuItem:
    def __init__(self, page):
        self._page = page

    def click(self, timeout=None):
        self._page._pending_download = self._page._menu_target
        self._page._menu_target = None


class _FakeDownload:
    def __init__(self, filename):
        self.suggested_filename = filename

    def save_as(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-bytes")


class _FakeDownloadInfo:
    def __init__(self, page):
        self._page = page

    @property
    def value(self):
        return _FakeDownload(self._page._pending_download.name)


class _FakeExpectDownload:
    def __init__(self, page):
        self._page = page

    def __enter__(self):
        return _FakeDownloadInfo(self._page)

    def __exit__(self, *exc_info):
        return False


class _FakeRow:
    def __init__(self, page, node):
        self._page = page
        self.node = node

    def locator(self, selector):
        if selector == '[data-automationid="field-LinkFilename"]':
            return _FakeTextCell(self.node.name)
        if selector == '[data-automationid="field-DocIcon"]':
            icons = [_FakeIcon("Yellow folder")] if self.node.is_folder else []
            return _FakeIconCell(icons)
        raise AssertionError(f"unexpected selector {selector!r}")

    def scroll_into_view_if_needed(self):
        pass

    def hover(self):
        pass

    def dblclick(self, timeout=None):
        self._page.path = [*self._page.path, self.node.name]

    def get_by_label(self, label):
        assert label == "Show more actions for this item"
        return _FakeLocatorList([_FakeMoreButton(self._page, self.node)])


class _FakeRowList:
    def __init__(self, nodes, page):
        self._rows = [None, *(_FakeRow(page, node) for node in nodes)]

    def count(self):
        return len(self._rows)

    def nth(self, i):
        return self._rows[i]


class _FakeAbsentContainer:
    """Stands in for `.odspSpartanList` when it isn't found -- exercised by
    the nested-folder test, which doesn't care about virtualization."""

    @property
    def first(self):
        return self

    def count(self):
        return 0


class _FakeKeyboard:
    def press(self, key):
        pass


class _FakeNestedFolderPage:
    """A tiny fake OneDrive listing page backed by an in-memory folder tree,
    supporting exactly the page/locator calls `_download_all_rows_via_browser`
    makes -- enough to prove recursion into sub-folders (and returning to
    the parent listing afterwards) works, without a real browser."""

    def __init__(self, tree):
        self._tree = tree
        self.path: list[str] = []
        self.keyboard = _FakeKeyboard()
        self._menu_target = None
        self._pending_download = None
        self.download_attempts: list[str] = []

    def _current_node(self):
        node = self._tree
        for name in self.path:
            node = next(c for c in node.children if c.name == name)
        return node

    @property
    def url(self):
        return "fake://" + "/".join(self.path)

    def get_by_role(self, role, name=None, exact=None):
        if role == "row":
            return _FakeRowList(self._current_node().children, self)
        if role == "menuitem":
            items = [_FakeMenuItem(self)] if self._menu_target is not None else []
            return _FakeLocatorList(items)
        raise AssertionError(f"unexpected role {role!r}")

    def locator(self, selector):
        assert selector == onedrive_ingest._LIST_SCROLL_CONTAINER_SELECTOR
        return _FakeAbsentContainer()

    def wait_for_timeout(self, ms):
        pass

    def wait_for_load_state(self, state, timeout=None):
        pass

    def goto(self, url, wait_until=None, timeout=None):
        assert url.startswith("fake://")
        rest = url[len("fake://") :]
        self.path = rest.split("/") if rest else []

    def expect_download(self, timeout=None):
        return _FakeExpectDownload(self)


def test_download_all_rows_via_browser_recurses_into_sub_folders(tmp_path: Path):
    tree = _Node(
        "root",
        True,
        [
            _Node("SubA", True, [_Node("file_a.pdf", False)]),
            _Node("file_root.pdf", False),
        ],
    )
    page = _FakeNestedFolderPage(tree)

    downloaded = onedrive_ingest._download_all_rows_via_browser(page, tmp_path)

    assert sorted(p.relative_to(tmp_path).as_posix() for p in downloaded) == [
        "SubA/file_a.pdf",
        "file_root.pdf",
    ]
    assert (tmp_path / "SubA" / "file_a.pdf").read_bytes() == b"fake-bytes"
    assert (tmp_path / "file_root.pdf").read_bytes() == b"fake-bytes"
    # Recursing back out must leave the listing where it started.
    assert page.path == []


def test_download_all_rows_via_browser_handles_multiple_nesting_levels(tmp_path: Path):
    tree = _Node(
        "root",
        True,
        [
            _Node(
                "1990 NIKAH",
                True,
                [_Node("Daerah Petaling Jaya", True, [_Node("scan1.pdf", False)])],
            ),
        ],
    )
    page = _FakeNestedFolderPage(tree)

    downloaded = onedrive_ingest._download_all_rows_via_browser(page, tmp_path)

    assert [p.relative_to(tmp_path).as_posix() for p in downloaded] == [
        "1990 NIKAH/Daerah Petaling Jaya/scan1.pdf",
    ]


def test_download_all_rows_via_browser_skips_files_already_on_disk(tmp_path: Path):
    """Simulates retrying a submission after a partial/interrupted pull --
    marriage-be reuses the same destination directory across a retry (see
    onedrive/service.py::run_onedrive_fetch, which only deletes it on
    success), so a file already downloaded from the prior attempt must not
    be re-fetched, while a still-missing sibling still gets downloaded."""
    tree = _Node(
        "root",
        True,
        [
            _Node("already_downloaded.pdf", False),
            _Node("still_missing.pdf", False),
        ],
    )
    page = _FakeNestedFolderPage(tree)
    (tmp_path / "already_downloaded.pdf").write_bytes(b"previously-downloaded-bytes")

    downloaded = onedrive_ingest._download_all_rows_via_browser(page, tmp_path)

    assert sorted(p.name for p in downloaded) == ["already_downloaded.pdf", "still_missing.pdf"]
    # The "..." menu (the first UI action taken to download a file) was
    # never opened for the file that already existed -- only for the one
    # actually missing.
    assert page.download_attempts == ["still_missing.pdf"]
    assert (tmp_path / "already_downloaded.pdf").read_bytes() == b"previously-downloaded-bytes"
    assert (tmp_path / "still_missing.pdf").read_bytes() == b"fake-bytes"
    assert not (tmp_path / "still_missing.pdf.part").exists()


def test_download_all_rows_via_browser_recursion_also_resumes_nested_folders(tmp_path: Path):
    """The resume behavior applies at every nesting level, not just the top
    of a share -- a file several folders deep that already exists locally
    is skipped too."""
    tree = _Node(
        "root",
        True,
        [_Node("1990 NIKAH", True, [_Node("already.pdf", False), _Node("missing.pdf", False)])],
    )
    page = _FakeNestedFolderPage(tree)
    nested_dir = tmp_path / "1990 NIKAH"
    nested_dir.mkdir()
    (nested_dir / "already.pdf").write_bytes(b"old-bytes")

    downloaded = onedrive_ingest._download_all_rows_via_browser(page, tmp_path)

    assert sorted(p.relative_to(tmp_path).as_posix() for p in downloaded) == [
        "1990 NIKAH/already.pdf",
        "1990 NIKAH/missing.pdf",
    ]
    assert page.download_attempts == ["missing.pdf"]
    assert (nested_dir / "already.pdf").read_bytes() == b"old-bytes"


def test_row_is_folder_detects_folder_icon_by_aria_label_substring():
    page = _FakeNestedFolderPage(_Node("root", True, []))
    folder_row = _FakeRow(page, _Node("Some Folder", True))
    file_row = _FakeRow(page, _Node("some_file.pdf", False))

    assert onedrive_ingest._row_is_folder(folder_row) is True
    assert onedrive_ingest._row_is_folder(file_row) is False


def test_next_unprocessed_row_scrolls_virtualized_container_to_find_all_rows():
    """Simulates OneDrive's virtualized listing: only a growing window of
    rows exists in the DOM at once, and the scroll container's own
    `evaluate` calls are what grow it -- confirmed empirically against a
    real 2,244-item folder (61 DOM rows at rest, more only after scrolling
    the list container, not the page)."""
    names = [f"file_{i}.pdf" for i in range(10)]

    class _Window:
        exposed = 3

    class _ScrollingRows:
        def count(self):
            return 1 + min(_Window.exposed, len(names))

        def nth(self, i):
            return _FakeTextCellRow(names[i - 1])

    class _FakeTextCellRow:
        def __init__(self, name):
            self._name = name

        def locator(self, selector):
            assert selector == '[data-automationid="field-LinkFilename"]'
            return _FakeTextCell(self._name)

    class _ScrollingContainer:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def evaluate(self, script):
            if "el.scrollTop = el.scrollTop" in script:
                _Window.exposed = min(_Window.exposed + 3, len(names))
                return None
            at_bottom = _Window.exposed >= len(names)
            return {
                "scrollTop": 100 if at_bottom else 0,
                "scrollHeight": 100,
                "clientHeight": 100 if at_bottom else 10,
            }

    class _FakePage:
        def get_by_role(self, role, name=None, exact=None):
            assert role == "row"
            return _ScrollingRows()

        def locator(self, selector):
            assert selector == onedrive_ingest._LIST_SCROLL_CONTAINER_SELECTOR
            return _ScrollingContainer()

        def wait_for_timeout(self, ms):
            pass

    page = _FakePage()
    seen: set[str] = set()
    discovered = []
    for _ in range(20):
        found = onedrive_ingest._next_unprocessed_row(page, seen)
        if found is None:
            break
        name, _row = found
        seen.add(name)
        discovered.append(name)

    assert discovered == names
    assert onedrive_ingest._next_unprocessed_row(page, seen) is None
