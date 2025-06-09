import json
import tempfile
from pathlib import Path
from typing import Dict

import soundfile as sf

from .asr_funasr import asr_transcribe
from .diarization_3dspeaker import diarize
from .merge_asr_diarization import merge_results
from .speaker_register import identify


def _extract_segment(wav_path: str, start: float, end: float, out_path: Path) -> None:
    """从音频中截取指定区间保存为新文件."""
    data, sr = sf.read(wav_path)
    st = int(start * sr)
    ed = int(end * sr)
    sf.write(out_path, data[st:ed], sr)


def process_audio(audio_path: str, hf_token: str | None = None) -> list[Dict]:
    """执行分轨、识别说话人并转写文本."""
    diar = diarize(audio_path, include_overlap=True, hf_token=hf_token)
    asr = asr_transcribe(audio_path)
    merged = merge_results(asr, diar)

    speaker_map: Dict[int, str] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for seg in merged:
            spk = seg.get("speaker", -1)
            if spk in speaker_map:
                continue
            seg_path = Path(tmpdir) / f"spk{spk}.wav"
            _extract_segment(audio_path, seg["start"], seg["end"], seg_path)
            res = identify(str(seg_path))
            speaker_map[spk] = res[0][0] if res else f"spk{spk}"

    for seg in merged:
        seg["speaker_name"] = speaker_map.get(seg["speaker"], f"spk{seg['speaker']}")
    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="完整语音处理流程")
    parser.add_argument("audio", help="输入音频路径")
    parser.add_argument("--token", help="HuggingFace 访问令牌", dest="token")
    args = parser.parse_args()
    result = process_audio(args.audio, args.token)
    print(json.dumps(result, ensure_ascii=False, indent=2))
