import json
import subprocess
import tempfile
from pathlib import Path


def diarize(audio_path: str, include_overlap: bool = False, hf_token: str | None = None,
            speaker_num: int | None = None) -> list[dict]:
    """使用3D-Speaker进行说话人分离，返回分段结果。"""
    script = Path(__file__).resolve().parent.parent / "3D-Speaker" / "speakerlab" / "bin" / "infer_diarization.py"
    if not script.exists():
        raise FileNotFoundError("infer_diarization.py not found, please clone 3D-Speaker repository")

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "python", str(script),
            "--wav", audio_path,
            "--out_dir", tmpdir,
            "--out_type", "json",
        ]
        if include_overlap:
            cmd.append("--include_overlap")
            if hf_token:
                cmd.extend(["--hf_access_token", hf_token])
        if speaker_num is not None:
            cmd.extend(["--speaker_num", str(speaker_num)])
        subprocess.run(cmd, check=True)
        out_file = Path(tmpdir) / (Path(audio_path).stem + ".json")
        with open(out_file, "r", encoding="utf-8") as f:
            diar = json.load(f)
        return list(diar.values())


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diarization_3dspeaker.py <audio>")
        sys.exit(1)
    res = diarize(sys.argv[1])
    print(json.dumps(res, ensure_ascii=False, indent=2))
