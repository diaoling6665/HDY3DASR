from typing import List, Dict


def merge_results(asr: List[Dict], diar: List[Dict]) -> List[Dict]:
    """根据时间戳对齐 ASR 结果与说话人分段。"""
    merged = []
    di_idx = 0
    for segment in asr:
        st, ed = segment.get("timestamp", [0, 0])
        text = segment.get("text", "")
        while di_idx < len(diar) and diar[di_idx]["end"] <= st:
            di_idx += 1
        speaker = diar[di_idx]["speaker"] if di_idx < len(diar) else -1
        merged.append({
            "start": st,
            "end": ed,
            "speaker": speaker,
            "text": text,
        })
    return merged


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) != 3:
        print("Usage: python merge_asr_diarization.py asr.json diar.json")
        sys.exit(1)
    asr = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    diar = json.load(open(sys.argv[2], "r", encoding="utf-8"))
    result = merge_results(asr, diar)
    print(json.dumps(result, ensure_ascii=False, indent=2))
