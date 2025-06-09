# HDY3DASR

本项目示例性地构建一个本地语音理解流程，包含以下模块：

- **ASR**：使用 FunASR 将音频转写为带时间戳的文本。
- **Diarization**：调用 3D-Speaker 进行说话人分离。
- **发言人注册与识别**：利用 3D-Speaker 提取并比对声纹。
- **ASR + 分段合并**：根据时间码合并识别文本与说话人信息。
- **临时命令行脚本**：`app.py` 提供交互式演示流程。

后续将继续完善情绪识别及 UI 部分。

## 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：请确认安装的 `datasets` 版本不低于 2.19.0，过旧的版本会导致
> `infer_diarization.py` 无法正常导入。

仓库中已包含 3D-Speaker 源码，无需额外克隆。

## 模型下载与配置

1. **ASR 模型**：在 `config.yaml` 的 `model_path` 字段填入
   `damo/speech_paraformer_asr_zh-cn-16k-common-vocab8404-pytorch`，首次运行
   `asr_funasr.py` 时会自动从 ModelScope 下载。
2. **说话人分离**：执行 `3D-Speaker/speakerlab/bin/infer_diarization.py` 时
   会下载默认模型，如需检测说话人重叠请准备 HuggingFace 访问令牌。
3. **说话人识别**：`speaker_register.py` 与 `full_pipeline.py` 使用模型
   `iic/speech_eres2netv2_sv_zh-cn_16k-common`，同样会在首次运行时下载。

## 使用示例

一键处理完整音频：

```bash
python src/full_pipeline.py path/to/audio.wav --token <HF_TOKEN>
```

或运行交互脚本：

```bash
python app.py
```

按提示选择需要的功能即可。
