# whisper-onnx-tensorrt
ONNX and TensorRT implementation of Whisper.

This repository has been reimplemented with ONNX and TensorRT using [zhuzilin/whisper-openvino](https://github.com/zhuzilin/whisper-openvino) as a reference.

Enables execution only with onnxruntime with CUDA and TensorRT Excecution Provider enabled, no need to install PyTorch or TensorFlow. All backend logic using PyTorch was rewritten to a Numpy/CuPy implementation from scratch.

## 1. Environment
Although it can run directly on the host PC, we strongly recommend the use of Docker to avoid breaking the environment.

1. Docker
2. NVIDIA GPU (VRAM 16 GB or more recommended)
3. onnx 1.13.1
4. onnxruntime-gpu 1.13.1 (TensorRT Execution Provider custom)
5. CUDA 11.8
6. cuDNN 8.9
7. TensorRT 8.5.3
8. onnx-tensorrt 8.5-GA
9. cupy v12.0.0
10. etc (See Dockerfile.xxx)

## 2. Converted Models
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/381_Whisper

## 3. Docker run
```bash
git clone https://github.com/PINTO0309/whisper-onnx-tensorrt.git && cd whisper-onnx-tensorrt
```
### 3-1. CUDA ver
```bash
docker run --rm -it --gpus all -v `pwd`:/workdir pinto0309/whisper-onnx-cuda
```
### 3-2. TensorRT ver
```bash
docker run --rm -it --gpus all -v `pwd`:/workdir pinto0309/whisper-onnx-tensorrt
```

## 4. Docker build
If you do not need to build the docker image by yourself, you do not need to perform this step.
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
- `--model` option
    ```
    tiny.en
    tiny
    base.en
    base
    small.en
    small
    medium.en
    medium
    large-v1
    large-v2
    ```
- command

    The onnx file is automatically downloaded when the sample is run. Note that `Decoder` is run in CUDA, not TensorRT, because the shape of all input tensors must be undefined.
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
- parameters
    ```
    usage: transcribe.py
        [-h]
        [--model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2}]
        [--output_dir OUTPUT_DIR]
        [--verbose VERBOSE]
        [--task {transcribe,translate}]
        [--language {af, am, ...}]
        [--temperature TEMPERATURE]
        [--best_of BEST_OF]
        [--beam_size BEAM_SIZE]
        [--patience PATIENCE]
        [--length_penalty LENGTH_PENALTY]
        [--suppress_tokens SUPPRESS_TOKENS]
        [--initial_prompt INITIAL_PROMPT]
        [--condition_on_previous_text CONDITION_ON_PREVIOUS_TEXT]
        [--temperature_increment_on_fallback TEMPERATURE_INCREMENT_ON_FALLBACK]
        [--compression_ratio_threshold COMPRESSION_RATIO_THRESHOLD]
        [--logprob_threshold LOGPROB_THRESHOLD]
        [--no_speech_threshold NO_SPEECH_THRESHOLD]
        audio [audio ...]

    positional arguments:
      audio
        audio file(s) to transcribe

    optional arguments:
      -h, --help
        show this help message and exit
      --model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2}
        name of the Whisper model to use
        (default: small)
      --output_dir OUTPUT_DIR, -o OUTPUT_DIR
        directory to save the outputs
        (default: .)
      --verbose VERBOSE
        whether to print out the progress and debug messages
        (default: True)
      --task {transcribe,translate}
        whether to perform X->X speech recognition ('transcribe') or
        X->English translation ('translate')
        (default: transcribe)
      --language {af, am, ...}
        language spoken in the audio, specify None to perform language detection
        (default: None)
      --temperature TEMPERATURE
        temperature to use for sampling
        (default: 0)
      --best_of BEST_OF
        number of candidates when sampling with non-zero temperature
        (default: 5)
      --beam_size BEAM_SIZE
        number of beams in beam search, only applicable when temperature is zero
        (default: 5)
      --patience PATIENCE
        optional patience value to use in beam decoding,
        as in https://arxiv.org/abs/2204.05424,
        the default (1.0) is equivalent to conventional beam search
        (default: None)
      --length_penalty LENGTH_PENALTY
        optional token length penalty coefficient (alpha) as in
        https://arxiv.org/abs/1609.08144, uses simple lengt normalization by default
        (default: None)
      --suppress_tokens SUPPRESS_TOKENS
        comma-separated list of token ids to suppress during sampling;
        '-1' will suppress most special characters except common punctuations
        (default: -1)
      --initial_prompt INITIAL_PROMPT
        optional text to provide as a prompt for the first window.
        (default: None)
      --condition_on_previous_text CONDITION_ON_PREVIOUS_TEXT
        if True, provide the previous output of the model as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes
        less prone to getting stuck in a failure loop
        (default: True)
      --temperature_increment_on_fallback TEMPERATURE_INCREMENT_ON_FALLBACK
        temperature to increase when falling back when the decoding fails to meet either of
        the thresholds below
        (default: 0.2)
      --compression_ratio_threshold COMPRESSION_RATIO_THRESHOLD
        if the gzip compression ratio is higher than this value, treat the decoding as failed
        (default: 2.4)
      --logprob_threshold LOGPROB_THRESHOLD
        if the average log probability is lower than this value, treat the decoding as failed
        (default: -1.0)
      --no_speech_threshold NO_SPEECH_THRESHOLD
        if the probability of the <|nospeech|> token is higher than this value AND
        the decoding has failed due to `logprob_threshold`, consider the segment as silence
        (default: 0.6)
    ``` 
## 6. Languages
```
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "iw": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}
```
