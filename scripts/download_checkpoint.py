"""Download the pretrained MAT Places512 checkpoint (fp16 safetensors mirror).

Weights are CC-BY-NC (research use only) — see https://github.com/fenglinglwb/MAT.
"""

import urllib.request
from pathlib import Path

URL = "https://huggingface.co/Acly/MAT/resolve/main/MAT_Places512_G_fp16.safetensors"
DEST = Path(__file__).resolve().parent.parent / "models" / "MAT_Places512_G_fp16.safetensors"

if __name__ == "__main__":
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print(f"already present: {DEST} ({DEST.stat().st_size} bytes)")
    else:
        print(f"downloading {URL} -> {DEST}")
        urllib.request.urlretrieve(URL, DEST)
        print(f"done ({DEST.stat().st_size} bytes)")
