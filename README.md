<div align="center">

<p align="center">
  <img src="assets/logo2.jpeg" alt="MultiTalk" width="240"/>
</p>

<h1>Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation</h1>


[Zhe Kong*](https://scholar.google.com/citations?user=4X3yLwsAAAAJ&hl=zh-CN) · [Feng Gao*](https://scholar.google.com/citations?user=lFkCeoYAAAAJ) ·[Yong Zhang](https://yzhang2016.github.io/)<sup>&#9993;</sup> · [Zhuoliang Kang](https://scholar.google.com/citations?user=W1ZXjMkAAAAJ&hl=en) · [Xiaoming Wei](https://scholar.google.com/citations?user=JXV5yrZxj5MC&hl=zh-CN) · [Xunliang Cai](https://openreview.net/profile?id=~Xunliang_Cai1)  

[Guanying Chen](https://guanyingc.github.io/) · [Wenhan Luo](https://whluo.github.io/)<sup>&#9993;</sup>

<sup>*</sup>Equal Contribution
<sup>&#9993;</sup>Corresponding Authors


<a href='https://meigen-ai.github.io/multi-talk/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2505.22647'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/MeiGen-AI/MeiGen-MultiTalk'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
</div>

> **TL; DR:**  MultiTalk is an audio-driven multi-person conversational video generation​​. It enables the video creation of multi-person conversation 💬, singing  🎤,  interaction control 👬, and cartoon 🙊.

<p align="center">
  <img src="assets/pipe.png">
</p>

## Video Demos

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/e55952e6-e1b2-44a5-9887-a89307a378da" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/f0396c19-d459-42aa-9d78-34fdea10de18" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/3576fd04-3e5f-4933-ac7b-1c4e6a601379" width="320" controls loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/5589056e-3202-442d-a62a-2cad7a7ecb19" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/554bfbe7-0090-492c-94be-329f5e39e175" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/9e961f35-9413-4846-a806-8186d54061da" width="320" controls loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/342595ab-cf75-4872-8182-f20fe8c95611" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/6476f9f0-35e0-4484-91a4-8aa646aa994a" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/d8fc8e94-0cba-4c25-9f3a-a8d7e0a785e1" width="320" controls loop></video>
     </td>
  </tr>
</table>





## ✨ Key Features

We propose **MultiTalk** , a novel framework for audio-driven multi-person conversational video generation. Given a multi-stream audio input, a reference image and a prompt, MultiTalk generates a video containing interactions following the prompt, with consistent lip motions aligned with the audio.

> - 💬 **​​Realistic Conversations**​​ - Support single & multi-person generation
> - 👥 ​​**​​Interactive Character Control**​​​​ - Direct virtual humans via prompts
> - 🎤 ​​**​​Generalization Performances**​​​​ - Support the generation of cartoon character and singing 
> - 📺 **​​​​Resolution Flexibility​​**​​: 480p & 720p output at arbitrary aspect ratios
> - ⏱️ **Long Video Generation**: Support video generation up to 15 seconds

## 🔥 Latest News

* June 9, 2025: 🔥🔥 We release the [weights](https://huggingface.co/MeiGen-AI/MeiGen-MultiTalk) and inference code of **MultiTalk** 
* May 29, 2025: We release the [Technique-Report](https://arxiv.org/abs/2505.22647) of **MultiTalk** 
* May 29, 2025: We release the [project page](https://meigen-ai.github.io/multi-talk/) of **MultiTalk** 

## 📑 Todo List

- [x] Release the technical report
- [x] Inference
- [x] Checkpoints
- [ ] TTS integration
- [ ] Multi-GPU Inference
- [ ] Gradio demo
- [ ] Inference acceleration


## Quick Start

### 🛠️Installation

#### 1. Create a conda environment and install pytorch, xformers
```
conda create -n multitalk python=3.10
conda activate multitalk
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121
```
#### 2. Flash-attn installation:
```
pip install packaging
pip install ninja
pip install psutil==7.0.0
pip install flash-attn==2.5.6 --no-build-isolation
```
or
```
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python -m pip install flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

#### 2. Other dependencies
```
conda install -c conda-forge librosa
pip install -r requirements.txt
```

#### 3. FFmeg installation
```
conda install -c conda-forge ffmpeg
```
or
```
sudo yum install ffmpeg ffmpeg-devel
```

### 🧱Model Preparation

#### 1. Model Download

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-I2V-14B-480P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)       | Base model
| chinese-wav2vec2-base |      🤗 [Huggingface](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base)          | Audio encoder
| MeiGen-MultiTalk      |      🤗 [Huggingface](https://huggingface.co/MeiGen-AI/MeiGen-MultiTalk)              | Our audio condition weights

Download models using huggingface-cli:
``` sh
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/MeiGen-MultiTalk --local-dir ./weights/MeiGen-MultiTalk
```

#### 2. Link MultiTalk Model to Wan2.1-I2V-14B-480P Directory
```
mv weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json_old
ln -s weights/MeiGen-MultiTalk/diffusion_pytorch_model.safetensors.index.json weights/Wan2.1-I2V-14B-480P/
ln -s weights/MeiGen-MultiTalk/multitalk.safetensors weights/Wan2.1-I2V-14B-480P/
```

### 🔑 Quick Inference

Our model is compatible with both 480P and 720P resolutions. The current code only supports 480P inference. 720P inference requires multiple GPUs, and we will provide an update soon.
> Some tips
> - Lip synchronization accuracy:​​ Audio CFG works optimally between 3–5. Increase the audio CFG value for better synchronization.
> - ​​Video clip length:​​ The model was trained on 81-frame videos at 25 FPS. For optimal prompt following performance, generate clips at 81 frames. Generating up to 201 frames is possible, though longer clips might reduce prompt-following performance.
> - ​​Long video generation:​​ Audio CFG influences color tone consistency across segments. Set this value to 3 to alleviate tonal variations.
> - Sampling steps: If you want to generate video faster, you can decrease the sampling step to even 10 that will not hurt the lip synchronization accuracy, but affects the motion quality. More sampling steps, better visual quality. 


#### 1. Single-Person

##### 1) Generate a short video with one chunk

```
python generate_multitalk.py --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' --input_json examples/single_example_1.json --sample_steps 40 --frame_num 81 --mode clip --save_file single_exp

```

##### 2) Long video generation

```
python generate_multitalk.py --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' --input_json examples/single_example_1.json --sample_steps 40 --mode streaming --save_file single_long_exp

```

#### 2. Multi-Person

###### 1) Generate a short video with one chunk

```
python generate_multitalk.py --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' --input_json examples/multitalk_example_1.json --sample_steps 40 --frame_num 81 --mode clip --save_file multi_exp
```

##### 2) Long video generation

```
python generate_multitalk.py --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' --input_json examples/multitalk_example_2.json --sample_steps 40 --mode streaming --save_file multi_long_exp

```


## 📚 Citation

If you find our work useful in your research, please consider citing:

```
@article{kong2025let,
  title={Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation},
  author={Kong, Zhe and Gao, Feng and Zhang, Yong and Kang, Zhuoliang and Wei, Xiaoming and Cai, Xunliang and Chen, Guanying and Luo, Wenhan},
  journal={arXiv preprint arXiv:2505.22647},
  year={2025}
}
```

## 📜 License
The models in this repository are licensed under the Apache 2.0 License. We claim no rights over the your generated contents, 
granting you the freedom to use them while ensuring that your usage complies with the provisions of this license. 
You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, 
causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations. 

