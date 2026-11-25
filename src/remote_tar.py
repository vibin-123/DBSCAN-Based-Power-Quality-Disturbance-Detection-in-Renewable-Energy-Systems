"""Random access to a remote, uncompressed tar archive over HTTP Range requests.

The SPARK dataset is published as one 76 GB ``.tar`` (BagIt) file. Downloading
all of it to use well under 1 % is wasteful, so this module walks the tar
headers with small ranged reads to build an index (member name, data offset,
size) and then fetches individual members by byte range.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterator

import requests

BLOCK = 512
MIN_INTERVAL = 1.0  # seconds between ranged requests


def _ranged_get(session: requests.Session, url: str, start: int, length: int,
                retries: int = 8) -> bytes:
    """Read ``length`` bytes at ``start``; retries with exponential backoff."""
    headers = {"Range": f"bytes={start}-{start + length - 1}"}
    for attempt in range(retries):
        time.sleep(MIN_INTERVAL)  # be polite to the repository
        try:
            r = session.get(url, headers=headers, timeout=60)
            if r.status_code == 206:
                return r.content
            if r.status_code == 416:  # past end of file
                return b""
            if r.status_code == 429:  # rate limited: honour Retry-After
                wait = int(r.headers.get("Retry-After", 0) or 0)
                time.sleep(max(wait, 20 * (attempt + 1)))
                continue
            raise IOError(f"HTTP {r.status_code}")
        except (requests.RequestException, IOError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise IOError(f"range request failed after {retries} attempts "
                  f"(bytes={start}-{start + length - 1})")


def _octal(field: bytes) -> int:
    if field and field[0] & 0x80:  # GNU base-256 encoding for big files
        return int.from_bytes(field[1:], "big")
    s = field.split(b"\0", 1)[0].strip()
    return int(s, 8) if s else 0


def _pax_fields(payload: bytes) -> dict[str, str]:
    out, pos = {}, 0
    while pos < len(payload):
        sp = payload.find(b" ", pos)
        if sp < 0:
            break
        try:
            n = int(payload[pos:sp])
        except ValueError:
            break
        rec = payload[sp + 1:pos + n - 1]
        if b"=" in rec:
            k, v = rec.split(b"=", 1)
            out[k.decode()] = v.decode("utf-8", "replace")
        pos += n
    return out


def iter_members(url: str, start_offset: int = 0, peek: int = 4096,
                 session: requests.Session | None = None
                 ) -> Iterator[tuple[str, int, int, int]]:
    """Yield ``(name, data_offset, size, next_header_offset)`` per regular file."""
    session = session or requests.Session()
    offset = start_offset
    pending_name: str | None = None
    pending_size: int | None = None
    buf, buf_start = b"", 0

    def read(at: int, n: int) -> bytes:
        nonlocal buf, buf_start
        if not (buf_start <= at and at + n <= buf_start + len(buf)):
            buf = _ranged_get(session, url, at, max(n, peek))
            buf_start = at
        return buf[at - buf_start: at - buf_start + n]

    while True:
        hdr = read(offset, BLOCK)
        if len(hdr) < BLOCK or hdr == b"\0" * BLOCK:
            return
        size = _octal(hdr[124:136])
        typeflag = hdr[156:157]
        name = hdr[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
        prefix = hdr[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
        if hdr[257:262] == b"ustar" and prefix:
            name = prefix + "/" + name
        data_offset = offset + BLOCK
        padded = (size + BLOCK - 1) // BLOCK * BLOCK
        if typeflag in (b"x", b"g"):          # pax extended header
            fields = _pax_fields(read(data_offset, size))
            if typeflag == b"x":
                pending_name = fields.get("path", pending_name)
                if "size" in fields:
                    pending_size = int(fields["size"])
        elif typeflag == b"L":               # GNU long name
            pending_name = read(data_offset, size).split(b"\0", 1)[0].decode(
                "utf-8", "replace")
        else:
            if pending_size is not None:
                size = pending_size
                padded = (size + BLOCK - 1) // BLOCK * BLOCK
            if typeflag in (b"0", b"\0", b""):
                yield (pending_name or name, data_offset, size,
                       data_offset + padded)
            pending_name = pending_size = None
        offset = data_offset + padded


def build_index(url: str, out_csv: Path, log_every: int = 1) -> Path:
    """Walk the whole archive and write ``name,offset,size`` rows (resumable)."""
    out_csv = Path(out_csv)
    start, mode = 0, "w"
    if out_csv.exists():
        rows = list(csv.reader(out_csv.open()))
        if len(rows) > 1:
            start, mode = int(rows[-1][3]), "a"
    with out_csv.open(mode, newline="") as fh:
        w = csv.writer(fh)
        if mode == "w":
            w.writerow(["name", "offset", "size", "next_header"])
        for i, row in enumerate(iter_members(url, start)):
            w.writerow(row)
            if i % log_every == 0:
                fh.flush()
                print(i, row[0], row[1], flush=True)
    return out_csv


def fetch_member(url: str, offset: int, size: int, dest: Path,
                 chunk: int = 8 << 20,
                 session: requests.Session | None = None) -> Path:
    """Download one archive member (by byte range) to ``dest``."""
    session = session or requests.Session()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as fh:
        pos = 0
        while pos < size:
            n = min(chunk, size - pos)
            fh.write(_ranged_get(session, url, offset + pos, n))
            pos += n
    tmp.rename(dest)
    return dest


if __name__ == "__main__":
    import sys
    build_index(sys.argv[1], Path(sys.argv[2]))


class HttpRangeFile:
    """Read-only, seekable file object over a byte window of a remote file.

    Lets :mod:`zipfile` open a zip that sits *inside* the remote tar without
    downloading it: only the central directory and the requested members are
    transferred. Reads are served from a block cache to limit request count.
    """

    def __init__(self, url: str, base: int, size: int, block: int = 32 << 20,
                 session: requests.Session | None = None):
        self.url, self.base, self.size, self.block = url, base, size, block
        self.session = session or requests.Session()
        self.pos = 0
        self._cache: dict[int, bytes] = {}
        self.requests_made = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else (
            self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._cache:
            start = i * self.block
            n = min(self.block, self.size - start)
            for attempt in range(5):   # a short block would truncate read()
                data = _ranged_get(self.session, self.url, self.base + start, n)
                self.requests_made += 1
                if len(data) == n:
                    break
                time.sleep(2 ** attempt)
            else:
                raise IOError(f"short read at block {i}: got {len(data)} "
                              f"of {n} bytes")
            self._cache[i] = data
            if len(self._cache) > 6:           # keep memory bounded
                self._cache.pop(next(iter(self._cache)))
        return self._cache[i]

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.block)
            piece = self._block(i)[o:o + n]
            if not piece:
                break
            out += piece
            self.pos += len(piece)
            n -= len(piece)
        return bytes(out)
