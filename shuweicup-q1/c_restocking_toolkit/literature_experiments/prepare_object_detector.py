"""Fetch only data and safe weights from a fixed Grounding DINO release."""
import hashlib
import time
from pathlib import Path

import requests


REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
REPO = "IDEA-Research/grounding-dino-tiny"
ROOT = Path(__file__).resolve().parents[1] / "cache/object_localization"
MODEL = ROOT / "grounding-dino-tiny"
PACKAGES = ROOT / "packages"
FILES = ("config.json", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.txt", "added_tokens.json", "model.safetensors")


def download():
    MODEL.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        response = session.get(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}", timeout=45)
        response.raise_for_status()
        metadata = {entry["path"]: entry for entry in response.json()}
        for name in FILES:
            info = metadata[name]
            dest = MODEL / name
            partial = dest.with_suffix(dest.suffix+".part")
            if dest.is_file() and dest.stat().st_size == info["size"]:
                continue
            for attempt in range(6):
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
                    with session.get(url, params={"download": "true", "attempt": attempt},
                                     headers={"Range": f"bytes={offset}-"} if offset else {},
                                     stream=True, timeout=(20, 75)) as payload:
                        payload.raise_for_status()
                        append = bool(offset and payload.status_code == 206)
                        if append and not payload.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Server returned an unexpected byte range.")
                        with partial.open("ab" if append else "wb") as output:
                            for block in payload.iter_content(1024*1024):
                                output.write(block)
                    if partial.stat().st_size != info["size"]:
                        raise OSError(f"Incomplete {name}, {partial.stat().st_size} of {info['size']} bytes")
                    with partial.open("rb") as f:
                        if "lfs" in info:
                            digest = hashlib.file_digest(f, "sha256").hexdigest()
                            expected = info["lfs"]["oid"]
                        else:
                            h = hashlib.sha1(f"blob {info['size']}\0".encode())
                            h.update(f.read())
                            digest, expected = h.hexdigest(), info["oid"]
                    if digest != expected:
                        raise ValueError(f"Integrity check failed for {name}")
                    partial.replace(dest)
                    print(name, info["size"], "verified", flush=True)
                    break
                except (OSError, requests.RequestException) as err:
                    print(f"{name}: attempt {attempt+1}, {type(err).__name__}", flush=True)
                    if attempt == 5:
                        raise
                    time.sleep(1+attempt)


if __name__ == "__main__":
    download()
