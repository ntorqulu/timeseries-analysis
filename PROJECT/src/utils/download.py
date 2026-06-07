"""
download.py — Download parquet files from the shared Google Drive folders.

Usage:
    python src/utils/download.py                  # from PROJECT/ directory
    python -m src.utils.download                  # from PROJECT/ directory

Options:
    python src/utils/download.py --tables trains timetables
    python src/utils/download.py --start 2026_03_14 --end 2026_03_20
    python src/utils/download.py --static-only
    python src/utils/download.py --dynamic-only

Authentication:
    First run triggers a browser OAuth flow; token is cached in token.json.
    Share credentials.json (OAuth client secrets) with team members, but never
    commit it.  token.json is user-specific — also gitignored.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Ensure the src directory is on sys.path so that `from utils.config import ...`
# works regardless of the working directory from which the script is invoked.
_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from utils.config import DATA_DIR, DRIVE_FOLDER, STATIC_DIR, DYNAMIC_DIR, DATE_FMT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCOPES           = ["https://www.googleapis.com/auth/drive.file"]
ROOT             = Path(__file__).parent.parent.parent
CREDENTIALS_PATH = ROOT / "credentials.json"
TOKEN_PATH       = ROOT / "token.json"

STATIC_TABLES  = ["stations", "lines"]
DYNAMIC_TABLES = ["trains", "timetables", "journeys", "weather"]


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_service():
    """Return an authenticated Drive v3 service, refreshing/creating token as needed."""
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Refreshing expired token …")
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDENTIALS_PATH}.\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
        log.info(f"Token saved to {TOKEN_PATH}")
    return build("drive", "v3", credentials=creds)


# ── Drive helpers ─────────────────────────────────────────────────────────────

def list_files(service, folder_id: str) -> list[dict]:
    """Return all non-trashed files in a Drive folder."""
    results = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, size, modifiedTime)",
                pageToken=page_token,
                pageSize=1000,
            )
            .execute()
        )
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def download_file(service, file_id: str, dest_path: Path) -> None:
    """Download a single Drive file to dest_path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request, chunksize=8 * 1024 * 1024)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dest_path.write_bytes(buf.getvalue())


# ── Download logic ────────────────────────────────────────────────────────────

def download_static(service, tables: list[str] | None = None) -> None:
    """Download stations.parquet and lines.parquet."""
    tables = tables or STATIC_TABLES
    for table in [t for t in tables if t in STATIC_TABLES]:
        folder_id = DRIVE_FOLDER[table]
        files = list_files(service, folder_id)
        if not files:
            log.warning(f"[{table}] no files found in Drive folder")
            continue
        # Static: only one file per table — always overwrite with latest
        latest = sorted(files, key=lambda f: f["modifiedTime"], reverse=True)[0]
        dest = STATIC_DIR / f"{table}.parquet"
        if dest.exists():
            log.info(f"[{table}] already exists, overwriting with latest from Drive")
        log.info(f"[{table}] downloading {latest['name']} …")
        download_file(service, latest["id"], dest)
        log.info(f"[{table}] saved to {dest}")


def download_dynamic(
    service,
    tables: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    skip_existing: bool = True,
) -> None:
    """
    Download daily parquet files for dynamic tables.

    Parameters
    ----------
    tables      : subset of DYNAMIC_TABLES; None → all
    start_date  : "YYYY_MM_DD" inclusive; None → all available
    end_date    : "YYYY_MM_DD" inclusive; None → all available
    skip_existing: if True, skip files already on disk (default)
    """
    tables = tables or DYNAMIC_TABLES
    tables = [t for t in tables if t in DYNAMIC_TABLES]

    # Build date filter set if range provided
    if start_date or end_date:
        sd = datetime.strptime(start_date, DATE_FMT) if start_date else datetime(2000, 1, 1)
        ed = datetime.strptime(end_date, DATE_FMT) if end_date else datetime.now()
        date_range = set()
        cur = sd
        while cur <= ed:
            date_range.add(cur.strftime(DATE_FMT))
            cur += timedelta(days=1)
    else:
        date_range = None

    for table in tables:
        folder_id = DRIVE_FOLDER[table]
        files = list_files(service, folder_id)
        log.info(f"[{table}] found {len(files)} file(s) in Drive")

        for f in sorted(files, key=lambda x: x["name"]):
            name: str = f["name"]
            if not name.endswith(".parquet"):
                continue

            # Extract date token from filename, e.g. "trains_2026_03_14.parquet"
            stem = name.replace(".parquet", "")
            parts = stem.split("_", 1)          # ["trains", "2026_03_14"]
            date_token = parts[1] if len(parts) == 2 else None

            if date_range is not None and date_token not in date_range:
                continue

            dest = DYNAMIC_DIR / table / name
            if skip_existing and dest.exists():
                log.debug(f"[{table}] {name} already on disk, skipping")
                continue

            log.info(f"[{table}] downloading {name} …")
            download_file(service, f["id"], dest)
            log.info(f"[{table}] saved to {dest}")


# ── Entry point ───────────────────────────────────────────────────────────────

def download_all(
    tables: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    static_only: bool = False,
    dynamic_only: bool = False,
    skip_existing: bool = True,
) -> None:
    """
    Main download entry point — call this from notebooks or CLI.

    Example (notebook):
        from src.download import download_all
        download_all(tables=["trains", "timetables"], start_date="2026_03_14")
    """
    service = get_service()

    if not dynamic_only:
        static_tables = [t for t in (tables or []) if t in STATIC_TABLES] or None
        download_static(service, tables=static_tables)

    if not static_only:
        dynamic_tables = [t for t in (tables or []) if t in DYNAMIC_TABLES] or None
        download_dynamic(
            service,
            tables=dynamic_tables,
            start_date=start_date,
            end_date=end_date,
            skip_existing=skip_existing,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Rodalies parquet data from Drive")
    parser.add_argument("--tables", nargs="+", help="tables to download (default: all)")
    parser.add_argument("--start", dest="start_date", help="start date YYYY_MM_DD (dynamic only)")
    parser.add_argument("--end", dest="end_date", help="end date YYYY_MM_DD (dynamic only)")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--dynamic-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-download even if file exists")
    args = parser.parse_args()

    download_all(
        tables=args.tables,
        start_date=args.start_date,
        end_date=args.end_date,
        static_only=args.static_only,
        dynamic_only=args.dynamic_only,
        skip_existing=not args.force,
    )