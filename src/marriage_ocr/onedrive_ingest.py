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
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_SCOPES = ["Files.Read.All"]
DEFAULT_AUTHORITY = "https://login.microsoftonline.com/common"


def encode_share_url(share_url: str) -> str:
    """Encode a sharing URL into the Graph API's `u!` shareId token.

    Per Microsoft's documented scheme (shares-get): base64-encode the URL,
    convert to unpadded base64url (`+` -> `-`, `/` -> `_`, strip trailing
    `=`), then prepend "u!".
    """
    raw = base64.b64encode(share_url.strip().encode("utf-8")).decode("ascii")
    unpadded = raw.rstrip("=").replace("/", "_").replace("+", "-")
    return f"u!{unpadded}"


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
