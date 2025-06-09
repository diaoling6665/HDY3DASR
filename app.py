import json
from pathlib import Path

from src.asr_funasr import asr_transcribe
from src.diarization_3dspeaker import diarize
from src.merge_asr_diarization import merge_results
from src.speaker_register import register, identify


def main():
    while True:
        print("\n请选择功能:")
        print("1) ASR 转写")
        print("2) 说话人分离")
        print("3) 合并 ASR 与分段")
        print("4) 注册发言人")
        print("5) 识别发言人")
        print("6) 退出")
        choice = input("输入编号: ").strip()

        if choice == "1":
            audio = input("请输入音频路径: ").strip()
            cfg = input("请输入配置文件路径(回车默认 config.yaml): ").strip() or "config.yaml"
            res = asr_transcribe(audio, cfg)
            print(json.dumps(res, ensure_ascii=False, indent=2))
        elif choice == "2":
            audio = input("请输入音频路径: ").strip()
            res = diarize(audio)
            print(json.dumps(res, ensure_ascii=False, indent=2))
        elif choice == "3":
            asr_file = input("请输入 ASR 结果 JSON 路径: ").strip()
            diar_file = input("请输入分段 JSON 路径: ").strip()
            with open(asr_file, 'r', encoding='utf-8') as f:
                asr_res = json.load(f)
            with open(diar_file, 'r', encoding='utf-8') as f:
                diar_res = json.load(f)
            merged = merge_results(asr_res, diar_res)
            print(json.dumps(merged, ensure_ascii=False, indent=2))
        elif choice == "4":
            name = input("请输入发言人姓名: ").strip()
            audio = input("请输入发言人音频路径: ").strip()
            register(name, audio)
            print("注册完成")
        elif choice == "5":
            audio = input("请输入待识别音频路径: ").strip()
            results = identify(audio)
            if results:
                for n, s in results:
                    print(f"{n}: {s:.4f}")
            else:
                print("未匹配到说话人")
        elif choice == "6":
            break
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()
