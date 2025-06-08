import json
import sys
from pathlib import Path

import yaml
from funasr import AutoModel


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def asr_transcribe(audio_path: str, config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    model_path = cfg.get("model_path")
    model = AutoModel.from_pretrained(model_path)
    results = model.generate(audio_path)
    output = []
    for r in results:
        output.append({
            "text": r.get("text", ""),
            "timestamp": r.get("timestamp", [])
        })
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python asr_funasr.py <audio_path> [config_path]")
        sys.exit(1)
    audio = sys.argv[1]
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    res = asr_transcribe(audio, cfg_path)
    print(json.dumps(res, ensure_ascii=False, indent=2))
