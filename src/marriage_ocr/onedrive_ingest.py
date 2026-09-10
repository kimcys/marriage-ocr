"""Pull an arbitrary OneDrive sharing link down to local disk.

The client "randomly dumps" a OneDrive share link per delivery -- there's no
fixed folder to sync, just whatever link shows up next. This resolves that
link via the Microsoft Graph API and mirrors its contents locally so the
existing `process`/`process-typed`/orchestrator commands can point `--input`
at a normal local folder as usual.

Important, easily-missed constraint (confirmed against current Microsoft
Graph documentation): the `/shares/{token}/driveItem` endpoint that resolves
a sharing URL to a DriveItem **requires a signed-in user (delegated auth)**.
Application-only permissions (a pure service-account style app registration)
are scoped to your own tenant's files and cannot resolve a link shared by an
external party at all, regardless of `Files.Read.All`. So this cannot be a
silent, fully unattended job -- it needs one real Microsoft identity
(personal or work/school, doesn't need to be in the client's tenant, same as
opening the link in a browser) authenticated once via the device-code flow
in `login()`. After that, the cached token refreshes silently.

`OneDriveClient` also needs an Entra ID app registration (a `client_id`) to
run the device-code flow at all. That registration lives in *your own*
tenant (every Microsoft account, personal or work/school, has one for free)
-- it never needs to be the client's tenant. If registering an app is
blocked entirely (org policy, no Entra ID access at all), `download_anonymous_share`
below is a zero-auth fallback: it works with a plain HTTP GET whenever the
client sets the share link's permission to "Anyone with the link", which is
the common case for one-off external drops. It cannot help with links
restricted to "People in your organization" or "Specific people" -- those
still redirect to a genuine Microsoft sign-in page no matter what, and only
the delegated `OneDriveClient` flow (or the client re-sharing as "Anyone")
gets past that.

A third case, confirmed empirically (not documented anywhere by Microsoft):
some "Anyone with the link" **folder** shares -- observed for folders that
have been migrated to a SharePoint Online backend -- respond to the plain
GET above with OneDrive's HTML web-app shell instead of raw file bytes or a
zip, even though the link never touches a sign-in host and the folder is
genuinely browsable anonymously. A real browser executing that page's JS
resolves the anonymous session and lists/downloads the files fine, so
`download_anonymous_share` falls back to a real headless-Chromium session
(`_download_via_browser`, via Playwright) whenever it hits exactly this
ambiguous shape (HTML body, non-sign-in host). This fallback is inherently
more fragile than the two paths above -- it depends on OneDrive's current
web UI markup, which Microsoft can change without notice -- so it's only
attempted when the plain GET can't be reasonably interpreted for the caller
some other way.
"""
from __future__ import annotations

import base64
import logging
import zipfile
from email.message import Message
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

LOGGER = logging.getLogger(__name__)

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_SCOPES = ["Files.Read.All"]
DEFAULT_AUTHORITY = "https://login.microsoftonline.com/common"

# Hosts a share link redirects to when it turns out to require sign-in
# (i.e. it wasn't actually shared as "Anyone with the link").
_SIGNIN_HOST_MARKERS = ("login.microsoftonline.com", "login.live.com", "login.windows.net")


def encode_share_url(share_url: str) -> str:
    """Encode a sharing URL into the Graph API's `u!` shareId token.

    Per Microsoft's documented scheme (shares-get): base64-encode the URL,
    convert to unpadded base64url (`+` -> `-`, `/` -> `_`, strip trailing
    `=`), then prepend "u!".
    """
    raw = base64.b64encode(share_url.strip().encode("utf-8")).decode("ascii")
    unpadded = raw.rstrip("=").replace("/", "_").replace("+", "-")
    return f"u!{unpadded}"


def _add_download_param(url: str) -> str:
    """Append `download=1`, which makes OneDrive/SharePoint stream the raw
    file (or a zip, for a folder link) instead of the HTML viewer page --
    but only for links shared as "Anyone with the link"."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query["download"] = "1"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _url_is_signin_host(url: str) -> bool:
    final_host = urlsplit(url).netloc.lower()
    return any(marker in final_host for marker in _SIGNIN_HOST_MARKERS)


def _filename_from_response(response: requests.Response, fallback: str) -> str:
    disposition = response.headers.get("Content-Disposition")
    if disposition:
        message = Message()
        message["Content-Disposition"] = disposition
        filename = message.get_filename()
        if filename:
            return filename
    name = urlsplit(response.url).path.rsplit("/", 1)[-1]
    return name or fallback


_SIGNIN_REQUIRED_MESSAGE = (
    "This OneDrive link requires signing in -- it isn't shared as "
    '"Anyone with the link". Ask the client to change the link\'s '
    'sharing permission to "Anyone", or use `onedrive login` + '
    "`onedrive fetch` instead (needs an Entra ID app registration)."
)


def download_anonymous_share(share_url: str, dest_dir: str | Path) -> list[Path]:
    """Download a OneDrive/SharePoint share link with a plain, unauthenticated
    HTTP GET -- no Entra ID app registration, no sign-in, no client-id.

    Only works when the client set the link's permission to "Anyone with the
    link"; anything scoped to "People in your organization" or "Specific
    people" redirects to a genuine Microsoft sign-in page, and this raises a
    `RuntimeError` telling the caller to fall back to `OneDriveClient`
    (`onedrive login` + `onedrive fetch`) or to ask the client to re-share
    the link as "Anyone".

    Handles both a single-file link and a folder link (a "Anyone" folder
    link comes back as a zip, which is extracted under `dest_dir`) -- and,
    for the folder shapes that come back as an HTML page instead of either
    of those despite not touching a sign-in host, falls back to a real
    headless-browser session (see the module docstring).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Resolve any redirects (e.g. a `1drv.ms` short link) first, without
    # reading the body, so `download=1` lands on the real sharepoint.com /
    # onedrive.live.com URL rather than the shortener.
    resolved = requests.get(share_url.strip(), timeout=60, stream=True, allow_redirects=True)
    resolved.close()
    resolved.raise_for_status()

    response = requests.get(
        _add_download_param(resolved.url), timeout=300, stream=True, allow_redirects=True
    )
    response.raise_for_status()

    if _url_is_signin_host(response.url):
        raise RuntimeError(_SIGNIN_REQUIRED_MESSAGE)

    if response.headers.get("Content-Type", "").startswith("text/html"):
        # Ambiguous: not a sign-in redirect, but not raw file/zip bytes
        # either -- some folder shares (observed for SharePoint-Online-
        # migrated folders) serve OneDrive's web-app shell here even though
        # they're genuinely browsable anonymously. A real browser executing
        # that page's JS can still pull the files out; see _download_via_browser.
        LOGGER.info(
            "Anonymous GET for %s returned an HTML page instead of raw file bytes; "
            "falling back to a headless-browser download.",
            share_url,
        )
        return _download_via_browser(share_url, dest_dir)

    filename = _filename_from_response(response, fallback="download")
    content_type = response.headers.get("Content-Type", "")
    is_zip = filename.lower().endswith(".zip") or "zip" in content_type

    if is_zip:
        # The zip itself always has to be downloaded in full -- OneDrive's
        # "download folder as zip" endpoint doesn't support resuming a
        # partial transfer -- but a caller retrying a previously-interrupted
        # attempt (marriage-be reuses the same `dest_dir` across a submission
        # retry; see onedrive/service.py::run_onedrive_fetch) can still skip
        # re-extracting whatever files a prior attempt already wrote out.
        buffer = BytesIO()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            buffer.write(chunk)
        buffer.seek(0)
        extracted: list[Path] = []
        with zipfile.ZipFile(buffer) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                target = dest_dir / info.filename
                if target.exists() and target.stat().st_size == info.file_size:
                    extracted.append(target)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Extract to a sibling temp path and rename into place only
                # once fully written, so a crash mid-extraction never leaves
                # a truncated file that the size check above would mistake
                # for a completed one on the next retry.
                tmp_target = target.with_name(target.name + ".part")
                with archive.open(info) as source, open(tmp_target, "wb") as handle:
                    handle.write(source.read())
                tmp_target.replace(target)
                extracted.append(target)
        return extracted

    local_path = dest_dir / filename
    with open(local_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            handle.write(chunk)
    return [local_path]


# Safety cap on how many items a single folder *listing* (one level) will
# process -- each file is a real UI-driven download, not a bulk zip (see
# _download_all_rows_via_browser), so an unbounded folder could otherwise
# hold a worker for a very long time. Applied independently at every level
# of the tree, not as one global budget across the whole share.
_MAX_BROWSER_DOWNLOAD_ITEMS = 300
_BROWSER_DOWNLOAD_TIMEOUT_MS = 60_000

# Safety cap on how deep `_download_all_rows_via_browser` will recurse into
# nested sub-folders, purely to bound a pathological/cyclic tree -- real
# client folders observed so far (category / district / year / files) only
# go three or four levels deep.
_MAX_BROWSER_FOLDER_DEPTH = 12

# The listing's own scroll container (observed empirically in OneDrive's
# current web UI: class name is auto-generated but always carries this
# stable "odspSpartanList" token). OneDrive virtualizes this list -- for a
# folder bigger than a screenful it keeps only a sliding window of ~30-60
# rows in the DOM and lazily grows `scrollHeight` as you scroll, rather than
# rendering every row up front. A single, un-scrolled `get_by_role("row")`
# snapshot therefore silently sees only the first screenful of a large
# folder (confirmed against a real 2,244-item folder: 61 DOM rows at rest,
# more only after scrolling the container, not the page).
_LIST_SCROLL_CONTAINER_SELECTOR = ".odspSpartanList"


def _row_name(row: Any, *, fallback: str) -> str:
    """The display name OneDrive shows for this row (file or folder)."""
    name_cell = row.locator('[data-automationid="field-LinkFilename"]')
    if name_cell.count() == 0:
        return fallback
    try:
        text = name_cell.inner_text().strip()
    except Exception:
        return fallback
    return text or fallback


def _row_is_folder(row: Any) -> bool:
    """True if `row` is a sub-folder rather than a file.

    Confirmed empirically: a folder row's icon cell has an `i[role="img"]`
    whose `aria-label` names the folder colour (e.g. "Yellow folder"); a
    file row instead has a plain `<img alt=".pdf">`-style icon with no such
    element. Checking for the substring "folder" (rather than an exact
    label) covers every folder colour OneDrive assigns, not just yellow.
    """
    icon_cell = row.locator('[data-automationid="field-DocIcon"]')
    icons = icon_cell.locator("i[role='img']")
    for i in range(icons.count()):
        label = icons.nth(i).get_attribute("aria-label") or ""
        if "folder" in label.lower():
            return True
    return False


def _next_unprocessed_row(page: Any, seen: set[str]) -> tuple[str, Any] | None:
    """One pass over the current folder listing, returning the first row
    whose name isn't in `seen` yet -- scrolling the virtualized list
    container forward (see `_LIST_SCROLL_CONTAINER_SELECTOR`) when the
    rows currently in the DOM are exhausted but the folder isn't. Returns
    None once nothing new turns up (folder fully listed, or scrolling twice
    in a row at the bottom yields no new row)."""
    container = page.locator(_LIST_SCROLL_CONTAINER_SELECTOR).first
    stable_passes_at_bottom = 0
    while True:
        rows = page.get_by_role("row")
        total = rows.count()
        for idx in range(1, total):
            row = rows.nth(idx)
            try:
                name = _row_name(row, fallback=f"item-{idx}")
            except Exception:
                continue
            if name not in seen:
                return name, row

        if container.count() == 0 or len(seen) >= _MAX_BROWSER_DOWNLOAD_ITEMS:
            return None

        try:
            metrics = container.evaluate(
                "el => ({scrollTop: el.scrollTop, scrollHeight: el.scrollHeight,"
                " clientHeight: el.clientHeight})"
            )
        except Exception:
            return None
        at_bottom = metrics["scrollTop"] + metrics["clientHeight"] >= metrics["scrollHeight"] - 2
        stable_passes_at_bottom = stable_passes_at_bottom + 1 if at_bottom else 0
        if stable_passes_at_bottom >= 2:
            return None

        container.evaluate("el => { el.scrollTop = el.scrollTop + el.clientHeight; }")
        page.wait_for_timeout(400)


def _download_all_rows_via_browser(page: Any, dest_dir: Path, *, _depth: int = 0) -> list[Path]:
    """Folder-listing shape: one row per file or sub-folder, each with its
    own "..." (Show more actions) menu. Downloads each file individually via
    that per-row menu's "Download" item -- not the shared multi-select
    toolbar, which empirically never fired a download event for a
    multi-file selection on this same page shape. A sub-folder row is
    entered (double-click navigates in, matching OneDrive's own
    `list-item-db-click` row action) and mirrored recursively into a
    same-named local sub-directory, so a share with categories/districts/
    years nested above the actual files gets fully walked, not just its
    top level.

    Resumable across retries: a file already present at its target path
    under `dest_dir` (from a prior call that got interrupted -- e.g. a
    timeout partway through a folder with thousands of items) is skipped
    rather than re-downloaded, since marriage-be reuses the same
    destination directory when retrying a failed OneDrive submission (see
    onedrive/service.py::run_onedrive_fetch, which only deletes it on
    success).
    """
    if _depth > _MAX_BROWSER_FOLDER_DEPTH:
        LOGGER.warning(
            "Folder nesting under %s exceeds %d levels; not descending further.",
            dest_dir,
            _MAX_BROWSER_FOLDER_DEPTH,
        )
        return []

    listing_url = page.url
    downloaded: list[Path] = []
    seen: set[str] = set()

    while len(seen) < _MAX_BROWSER_DOWNLOAD_ITEMS:
        found = _next_unprocessed_row(page, seen)
        if found is None:
            break
        name, row = found
        seen.add(name)
        try:
            # A prior row's click/selection can leave a stray overlay
            # ("SelectionZone") intercepting pointer events on the next
            # row -- clearing it before every row (not just after a
            # failure) is what actually made this reliable past the first
            # few rows in practice.
            page.keyboard.press("Escape")
            row.scroll_into_view_if_needed()
            row.hover()

            if _row_is_folder(row):
                row.dblclick(timeout=10_000)
                page.wait_for_load_state("networkidle", timeout=30_000)
                page.wait_for_timeout(1000)  # let the new listing's rows settle
                downloaded.extend(
                    _download_all_rows_via_browser(page, dest_dir / name, _depth=_depth + 1)
                )
                # Re-fetching the row list (rather than trying to resume the
                # one we navigated away from) after returning from a
                # sub-folder resets the virtualized listing to the top --
                # `seen` is what lets `_next_unprocessed_row` fast-forward
                # past everything already handled instead of redoing it.
                page.goto(listing_url, wait_until="networkidle", timeout=30_000)
                page.wait_for_timeout(1000)
                continue

            target = dest_dir / name
            if target.exists() and target.stat().st_size > 0:
                # Resuming a previously-interrupted pull of this same folder
                # (marriage-be reuses the same destination dir across a
                # submission retry; see onedrive/service.py::
                # run_onedrive_fetch) -- this file was already downloaded,
                # so skip re-triggering the browser download for it. Safe
                # because a file only ever exists at `target` once fully
                # written (see the temp-path rename below), never mid-write.
                downloaded.append(target)
                continue

            more_button = row.get_by_label("More Actions", exact=True)
            if more_button.count() == 0:
                continue
            more_button.first.click(timeout=10_000)

            menu_item = page.get_by_role("menuitem", name="Download", exact=False)
            if menu_item.count() == 0:
                continue

            with page.expect_download(timeout=_BROWSER_DOWNLOAD_TIMEOUT_MS) as dl_info:
                menu_item.first.click(timeout=10_000)
            download = dl_info.value
            dest_dir.mkdir(parents=True, exist_ok=True)
            # Named after the row's own display name (known upfront, and
            # what the exists-check above keys on) rather than
            # `download.suggested_filename` -- keeps the pre-download skip
            # check and the actual write pointed at the exact same path.
            tmp_target = dest_dir / f"{name}.part"
            download.save_as(tmp_target)
            tmp_target.replace(target)
            downloaded.append(target)
        except Exception:
            # One stuck/overlay-covered row shouldn't sacrifice every other
            # item in the folder -- log and move on to the next row.
            LOGGER.exception("Browser download failed for folder row %r; skipping it", name)
            continue

    if len(seen) >= _MAX_BROWSER_DOWNLOAD_ITEMS:
        LOGGER.warning(
            "Folder listing at %s has more than %d items; only processing the first %d.",
            dest_dir,
            _MAX_BROWSER_DOWNLOAD_ITEMS,
            _MAX_BROWSER_DOWNLOAD_ITEMS,
        )

    return downloaded


def _download_single_item_via_browser(page: Any, dest_dir: Path) -> list[Path]:
    """Single-file-preview shape: no row listing, just a page-level Download
    control in the toolbar."""
    control = page.get_by_role("button", name="Download", exact=False)
    if control.count() == 0:
        return []
    with page.expect_download(timeout=_BROWSER_DOWNLOAD_TIMEOUT_MS) as dl_info:
        control.first.click()
    download = dl_info.value
    target = dest_dir / download.suggested_filename
    download.save_as(target)
    return [target]


def _download_via_browser(share_url: str, dest_dir: Path) -> list[Path]:
    """Render `share_url` in a real headless browser and pull its files out
    through OneDrive's own web UI, for the ambiguous "HTML but not a sign-in
    redirect" case `download_anonymous_share` can't resolve with a plain GET.

    Empirically fragile by nature (depends on OneDrive's current web UI
    markup, which Microsoft can change without notice) -- prefer asking the
    client for a plain file-level "Anyone" link, or `onedrive login` +
    `onedrive fetch`, when either is available.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The anonymous GET for this OneDrive link returned an HTML page instead "
            "of raw file bytes (common for folder links migrated to SharePoint "
            "Online), so falling back to a real browser. That needs Playwright: "
            "pip install playwright && playwright install --with-deps chromium"
        ) from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            # No sandbox: containers rarely grant the user-namespace/seccomp
            # privileges Chromium's own sandbox wants, and this process
            # already only ever renders one trusted, read-only OneDrive page.
            args=["--no-sandbox"],
        )
        try:
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()
            page.goto(share_url, wait_until="networkidle", timeout=60_000)
            page.wait_for_timeout(2000)  # let client-side rendering settle

            if _url_is_signin_host(page.url):
                raise RuntimeError(_SIGNIN_REQUIRED_MESSAGE)

            downloaded = _download_all_rows_via_browser(page, dest_dir)
            if not downloaded:
                downloaded = _download_single_item_via_browser(page, dest_dir)
            if not downloaded:
                raise RuntimeError(
                    "OneDrive rendered a page that isn't a recognizable file or "
                    "folder listing -- the link may need a different sharing "
                    "permission, or Microsoft has changed their web UI in a way "
                    "this fallback doesn't yet handle."
                )
            return downloaded
        finally:
            browser.close()


class OneDriveClient:
    def __init__(
        self,
        *,
        client_id: str,
        token_cache_path: str | Path,
        authority: str = DEFAULT_AUTHORITY,
        scopes: list[str] | None = None,
    ) -> None:
        try:
            import msal
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OneDrive ingestion requires the `msal` package. Install it with: pip install msal"
            ) from exc

        self._scopes = scopes or DEFAULT_SCOPES
        self._token_cache_path = Path(token_cache_path)
        self._cache = msal.SerializableTokenCache()
        if self._token_cache_path.exists():
            self._cache.deserialize(self._token_cache_path.read_text(encoding="utf-8"))

        self._app = msal.PublicClientApplication(
            client_id, authority=authority, token_cache=self._cache
        )

    def login(self) -> None:
        """Interactive device-code login -- run this once yourself (the
        message it prints has a URL + code to enter in a browser); every
        later call to this class reuses and silently refreshes the cached
        token instead of prompting again."""
        flow = self._app.initiate_device_flow(scopes=self._scopes)
        if "user_code" not in flow:
            raise RuntimeError(f"Failed to start device login flow: {flow}")
        print(flow["message"])
        result = self._app.acquire_token_by_device_flow(flow)
        if "access_token" not in result:
            raise RuntimeError(f"OneDrive login failed: {result.get('error_description', result)}")
        self._save_cache()

    def download_share(self, share_url: str, dest_dir: str | Path) -> list[Path]:
        """Resolve `share_url` and mirror its full contents under `dest_dir`,
        preserving folder structure. Skips a file if a same-size local copy
        already exists, so a partially-completed pull can resume."""
        root = self.resolve_share(share_url)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        self._download_item_tree(root, dest_dir, downloaded)
        return downloaded

    def resolve_share(self, share_url: str) -> dict[str, Any]:
        token = encode_share_url(share_url)
        return self._get(f"{GRAPH_BASE_URL}/shares/{token}/driveItem")

    def _download_item_tree(self, item: dict[str, Any], local_dir: Path, downloaded: list[Path]) -> None:
        if "folder" in item:
            drive_id = item["parentReference"]["driveId"]
            item_id = item["id"]
            url: str | None = f"{GRAPH_BASE_URL}/drives/{drive_id}/items/{item_id}/children"
            while url:
                page = self._get(url)
                for child in page.get("value", []):
                    if "folder" in child:
                        self._download_item_tree(child, local_dir / child["name"], downloaded)
                    else:
                        downloaded.append(self._download_file(child, local_dir))
                url = page.get("@odata.nextLink")
            return

        downloaded.append(self._download_file(item, local_dir))

    def _download_file(self, item: dict[str, Any], local_dir: Path) -> Path:
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / item["name"]
        remote_size = item.get("size")
        if local_path.exists() and remote_size is not None and local_path.stat().st_size == remote_size:
            LOGGER.info("Skipping already-downloaded file %s", local_path)
            return local_path

        download_url = item.get("@microsoft.graph.downloadUrl")
        if download_url is None:
            raise RuntimeError(f"No download URL for item {item.get('name')!r}")

        # The download URL is itself pre-authenticated (short-lived, signed) --
        # a plain GET, no Authorization header needed or wanted here.
        response = requests.get(download_url, timeout=300, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                handle.write(chunk)
        return local_path

    def _access_token(self) -> str:
        accounts = self._app.get_accounts()
        if accounts:
            result = self._app.acquire_token_silent(self._scopes, account=accounts[0])
            if result and "access_token" in result:
                self._save_cache()
                return result["access_token"]
        raise RuntimeError(
            "No cached OneDrive login found (or it expired/was revoked). "
            "Run `marriage-ocr onedrive login` first."
        )

    def _get(self, url: str) -> dict[str, Any]:
        response = requests.get(
            url, headers={"Authorization": f"Bearer {self._access_token()}"}, timeout=60
        )
        response.raise_for_status()
        return response.json()

    def _save_cache(self) -> None:
        if self._cache.has_state_changed:
            self._token_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_cache_path.write_text(self._cache.serialize(), encoding="utf-8")
