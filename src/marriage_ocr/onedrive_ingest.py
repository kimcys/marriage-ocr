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
still redirect to a Microsoft sign-in page no matter what, and only the
delegated `OneDriveClient` flow (or the client re-sharing as "Anyone") gets
past that.
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


def _looks_like_signin_redirect(response: requests.Response) -> bool:
    final_host = urlsplit(response.url).netloc.lower()
    if any(marker in final_host for marker in _SIGNIN_HOST_MARKERS):
        return True
    return response.headers.get("Content-Type", "").startswith("text/html")


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


def download_anonymous_share(share_url: str, dest_dir: str | Path) -> list[Path]:
    """Download a OneDrive/SharePoint share link with a plain, unauthenticated
    HTTP GET -- no Entra ID app registration, no sign-in, no client-id.

    Only works when the client set the link's permission to "Anyone with the
    link"; anything scoped to "People in your organization" or "Specific
    people" redirects to a Microsoft sign-in page, and this raises a
    `RuntimeError` telling the caller to fall back to `OneDriveClient`
    (`onedrive login` + `onedrive fetch`) or to ask the client to re-share
    the link as "Anyone".

    Handles both a single-file link and a folder link (a "Anyone" folder
    link comes back as a zip, which is extracted under `dest_dir`).
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

    if _looks_like_signin_redirect(response):
        raise RuntimeError(
            "This OneDrive link requires signing in -- it isn't shared as "
            '"Anyone with the link". Ask the client to change the link\'s '
            'sharing permission to "Anyone", or use `onedrive login` + '
            "`onedrive fetch` instead (needs an Entra ID app registration)."
        )

    filename = _filename_from_response(response, fallback="download")
    content_type = response.headers.get("Content-Type", "")
    is_zip = filename.lower().endswith(".zip") or "zip" in content_type

    if is_zip:
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
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, open(target, "wb") as handle:
                    handle.write(source.read())
                extracted.append(target)
        return extracted

    local_path = dest_dir / filename
    with open(local_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            handle.write(chunk)
    return [local_path]


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
