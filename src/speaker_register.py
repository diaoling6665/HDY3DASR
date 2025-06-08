import json
import subprocess
from pathlib import Path

DB_FILE = Path("speaker_db.json")
MODEL_ID = "iic/speech_campplus_sv_zh_en_16k-common_advanced"


def _load_db() -> dict:
    if DB_FILE.exists():
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_db(db: dict):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def register(name: str, audio_path: str):
    """注册发言人音频并保存其声纹。"""
    script = Path(__file__).resolve().parent.parent / "3D-Speaker" / "speakerlab" / "bin" / "infer_sv.py"
    if not script.exists():
        raise FileNotFoundError("infer_sv.py not found, please clone 3D-Speaker repository")

    out_dir = Path("registered")
    out_dir.mkdir(exist_ok=True)
    emb_path = out_dir / f"{name}.npy"
    cmd = [
        "python", str(script),
        "--model_id", MODEL_ID,
        "--wavs", audio_path,
        "--local_model_dir", str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    # embedding saved under out_dir/<model_id>/embeddings/*.npy
    model_dir = out_dir / MODEL_ID.split("/")[1] / "embeddings"
    files = list(model_dir.glob("*.npy"))
    if not files:
        raise RuntimeError("Embedding not generated")
    files[0].rename(emb_path)
    db = _load_db()
    db[name] = str(emb_path)
    _save_db(db)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python speaker_register.py <name> <audio>")
        sys.exit(1)
    register(sys.argv[1], sys.argv[2])
