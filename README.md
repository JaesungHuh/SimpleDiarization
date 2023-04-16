# SimpleDiarization

A simple diarization module using [pyannote](https://github.com/pyannote/pyannote-audio) voice activity detection and [speechbrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) speaker embedding extractor.

## Installation
``` 
git clone https://github.com/JaesungHuh/SimpleDiarization.git --recursive
```

We recommend to create conda environment with python version >= 3.9.

```
conda create -n simple python=3.9
conda activate simple
pip install -r requirements.txt
```

### If installation not working...
Please install the packages using the instructions from official websites.
- [pyyaml](https://pyyaml.org/)
- [pytorch](https://pytorch.org/)
- [speechbrain](https://speechbrain.github.io/)
- [pyannote](https://pyannote.github.io/)
- [sklearn](https://scikit-learn.org/stable/install.html)
- [tqdm](https://github.com/tqdm/tqdm)

## Configuration file
You need to change the configuration file for your own use. Please refer to [config.yaml](conf/config.yaml).
```
misc:
  input_list: "trials/voxconverse_dev_small.list"
  output_dir: "trials/voxconverse_dev"
  device: "cuda"
audio:
  sr: 16000
vad:
  ref_vad: false
  merge_vad: false
  ref_suffix: ".lab"
  pyannote_token: [YOUR PYANNOTE TOKEN]
  onset: 0.5
  offset: 0.5
  min_duration_on: 0.1
  min_duration_off: 0.1
embedding:
  win_length: 1.5
  hop_length: 0.5
cluster:
  num_cluster: None
  threshold: 0.8
  normalize: false
eval:
  run: true
  collar: 0.25
  ignore_overlaps: false
```

OVERALL DER : 5.65% JER : 17.15%
OVERALL DER : 7.99% JER : 25.13%