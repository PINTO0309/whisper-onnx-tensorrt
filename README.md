# whisper-onnx-tensorrt
ONNX and TensorRT implementation of Whisper.

This repository has been reimplemented with ONNX and TensorRT using [zhuzilin/whisper-openvino](https://github.com/zhuzilin/whisper-openvino) as a reference.

## 1. Environment
1. Docker
2. NVIDIA GPU

## 2. Converted Models
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/381_Whisper

## 3. Docker run
### 3-1. CUDA ver
```bash
docker run --rm -it --gpus all -v `pwd`:/workdir pinto0309/whisper-onnx-cuda
```
### 3-2. TensorRT ver
```bash
docker run --rm -it --gpus all -v `pwd`:/workdir pinto0309/whisper-onnx-tensorrt
```

## 4. Docker build
### 4-1. CUDA ver
```bash
docker build -t whisper-onnx -f Dockerfile.gpu .
```
### 4-2. TensorRT ver
```bash
docker build -t whisper-onnx -f Dockerfile.tensorrt .
```
### 4-3. docker run
```bash
docker run --rm -it --gpus all -v `pwd`:/workdir whisper-onnx
```

## 5. Transcribe
- command
    ```bash
    python whisper/transcribe.py xxxx.mp4 --model small --beam_size 3
    ```
- results
    ```
    Detecting language using up to the first 30 seconds. Use `--language` to specify the language
    Detected language: Japanese
    [00:00.000 --> 00:07.200] ストレオシンの推定モデルの最適化 としまして 後半のパート2は 実際
    [00:07.200 --> 00:11.600] のデモを交えまして 普段私がどのように モデルを最適化して 様々な
    [00:11.600 --> 00:15.600] フレームワークの環境でプロイしてる かというのを実際に操作をこの
    [00:15.600 --> 00:18.280] 画面上で見ていただきながら ご理解いただけるように努めたい
    [00:18.280 --> 00:21.600] と思います それでは早速ですが こちらの
    [00:21.600 --> 00:26.320] GitHubの方に本日の公演内容について は すべてチュートリアルをまとめて
    [00:26.320 --> 00:31.680] コミットしております 2021.0.20.28 インテルティブラーニング
    [00:31.680 --> 00:35.200] でヒットネットデモという ちょっと長い名前なんですけれども 現状
    [00:35.200 --> 00:39.120] はプライベートになってますが この公演のタイミングでパブリック
    [00:39.120 --> 00:43.440] の方に変更したいと思っております 基本的にはこちらの上から順前
    [00:43.440 --> 00:48.000] ですね チュートリアルを謎って いくという形になります
    [00:48.000 --> 00:52.640] まず本日対象にするモデルの内容 なんですけれども Google Research
    ```
