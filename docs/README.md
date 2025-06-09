# 项目文档

此目录用于存放自动生成的说明文档。当前阶段主要描述语音处理流程，包括：

1. 使用 `asr_funasr.py` 转写音频；
2. 通过 `diarization_3dspeaker.py` 进行说话人分离；
3. 使用 `merge_asr_diarization.py` 合并文字与说话人片段；
 k1spj6-codex/初始化项目结构与asr模块开发
4. `speaker_register.py` 提供声纹注册功能；
5. `app.py` 为临时命令行脚本，可交互式执行以上流程。
=======
4. `speaker_register.py` 提供声纹注册功能。
 main
