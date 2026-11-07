import hashlib
import json

import pytest

from a2ui_model_tools import download
from a2ui_model_tools.checkpoint_io import IdentityError
from a2ui_model_tools.download import digest_verifier, download_plan, fetch, wasm_provenance
from a2ui_model_tools.workdir import WorkDir

PAYLOAD = b"pinned bytes"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def read(self, size=-1):
        chunk, self.body = (self.body, b"") if size < 0 else (self.body[:size], self.body[size:])
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture
def serve(monkeypatch):
    """Answer urlopen with the given bodies in order."""
    calls = []

    def install(*bodies):
        queue = list(bodies)

        def urlopen(request, timeout):
            calls.append(request.full_url)
            return FakeResponse(queue.pop(0))

        monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(download.time, "sleep", lambda _: None)
        return calls

    return install


def test_downloads_and_verifies(tmp_path, serve):
    calls = serve(PAYLOAD)
    target = tmp_path / "dir" / "file"
    assert fetch("https://example.test/file", target, digest_verifier(PAYLOAD_SHA))
    assert target.read_bytes() == PAYLOAD
    assert calls == ["https://example.test/file"]
    assert not (tmp_path / "dir" / "file.download").exists()


def test_retries_a_corrupt_download(tmp_path, serve):
    calls = serve(b"corrupt", PAYLOAD)
    assert fetch("https://example.test/file", tmp_path / "file", digest_verifier(PAYLOAD_SHA))
    assert len(calls) == 2


def test_gives_up_without_leaving_files(tmp_path, serve):
    serve(b"corrupt", b"corrupt")
    with pytest.raises(IdentityError):
        fetch("https://example.test/file", tmp_path / "file", digest_verifier(PAYLOAD_SHA), attempts=2)
    assert list(tmp_path.iterdir()) == []


def test_keeps_a_verified_existing_file(tmp_path, serve):
    calls = serve()
    (tmp_path / "file").write_bytes(PAYLOAD)
    assert not fetch("https://example.test/file", tmp_path / "file", digest_verifier(PAYLOAD_SHA))
    assert calls == []


def test_never_overwrites_a_mismatching_file(tmp_path, serve):
    serve(PAYLOAD)
    (tmp_path / "file").write_bytes(b"local edit")
    with pytest.raises(IdentityError, match="pinned identity"):
        fetch("https://example.test/file", tmp_path / "file", digest_verifier(PAYLOAD_SHA))
    assert (tmp_path / "file").read_bytes() == b"local edit"


def test_size_is_checked(tmp_path):
    (tmp_path / "file").write_bytes(PAYLOAD)
    with pytest.raises(IdentityError, match="Size mismatch"):
        digest_verifier(PAYLOAD_SHA, size=1)(tmp_path / "file")


def test_plan_pins_every_file(tmp_path):
    work = WorkDir(tmp_path)
    plan = download_plan(work)
    targets = [target for _, target, _ in plan]
    assert len(targets) == len(set(targets))
    assert work.base / "model.safetensors" in targets
    assert work.adapter / "adapters.safetensors" in targets
    assert work.wasm in targets
    assert all(url.startswith("https://") for url, _, _ in plan)
    assert "/resolve/d36bdd3855c82a6a1a23f7c459b749f5724ae75d/" in plan[0][0]


def test_wasm_provenance_matches_pins():
    record = wasm_provenance()
    assert record["sha256"] == "6e9cdce823ca2872184146c9b650f0dcfcd3fb114d79c2e34afa4c20f73ce4d2"
    assert record["url"].endswith(record["path"])
    json.dumps(record)
