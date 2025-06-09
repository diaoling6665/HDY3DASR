# 项目文档

本目录记录各脚本的作用与使用方法：

1. `asr_funasr.py`：转写音频获取文字。
2. `diarization_3dspeaker.py`：进行说话人分离。
3. `merge_asr_diarization.py`：合并转写结果与分段信息。
4. `speaker_register.py`：注册或识别发言人声纹。
5. `full_pipeline.py`：自动完成分轨、识别和转写。
6. `app.py`：简单交互式演示脚本。
if0xeo-codex/实现多人中文语音对话的说话人分轨与识别

请确保安装依赖时 `datasets>=2.19.0`，否则 `infer_diarization.py` 可能因
导入失败而无法运行。

