import json
import subprocess
import tempfile

import sys

from pathlib import Path
from typing import List, Tuple

import numpy as np

DB_FILE = Path("speaker_db.json")
MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"


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
        sys.executable, str(script),
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


def identify(audio_path: str, top_k: int = 1) -> List[Tuple[str, float]]:
    """识别音频中的说话人，返回按相似度排序的候选列表。"""
    script = Path(__file__).resolve().parent.parent / "3D-Speaker" / "speakerlab" / "bin" / "infer_sv.py"
    if not script.exists():
        raise FileNotFoundError("infer_sv.py not found, please clone 3D-Speaker repository")

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [

            sys.executable, str(script),


            "--model_id", MODEL_ID,
            "--wavs", audio_path,
            "--local_model_dir", tmpdir,
        ]
        subprocess.run(cmd, check=True)
        emb_file = Path(tmpdir) / MODEL_ID.split("/")[1] / "embeddings" / f"{Path(audio_path).stem}.npy"
        if not emb_file.exists():
            raise RuntimeError("Embedding not generated")
        query_emb = np.load(emb_file)

    db = _load_db()
    results = []
    for name, path in db.items():
        ref_emb = np.load(path)
        score = float(np.dot(query_emb, ref_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(ref_emb) + 1e-8))
        results.append((name, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "identify":
        result = identify(sys.argv[2])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif len(sys.argv) == 3:
        register(sys.argv[1], sys.argv[2])
    else:
        print(
            "Usage:\n"
            "  python speaker_register.py <name> <audio>      # 注册\n"
            "  python speaker_register.py identify <audio>    # 识别"
        )
