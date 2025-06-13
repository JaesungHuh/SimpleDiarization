# SimpleDiarization

A simple diarization module using [pyannote](https://huggingface.co/pyannote/segmentation) voice activity detection, [speechbrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) speaker embedding extractor and AHC clustering.

## Installation
``` 
git clone https://github.com/JaesungHuh/SimpleDiarization.git --recursive
```

We recommend to create conda environment with python version >= 3.9.

```
conda create -n simple python=3.11
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

## How to use this

```bash
python main.py --cfg_file CONFIG_FILE
```
## Note
- This module currently only supports the diarization with single-channel, 16kHz, PCM_16 audio files. You may experience performance degradation if you process the audio files with other sampling rates. We advise you to run the following command before you run this module.
```
ffmpeg -i INPUT_AUDIO -acodec pcm_s16le -ac 1 -ar 16000 OUT_AUDIO
```

## Configuration file
You need to change the configuration file for your own use. Please refer to [config.yaml](conf/config.yaml).
```
misc:
  input_list: "data/example.list"
  output_dir: "data/example_result"
  device: "cuda"
audio:
  sr: 16000
vad:
  ref_vad: false
  merge_vad: false
  ref_suffix: ".lab"
  pyannote_token: PUT YOUR PYANNOTE TOKEN IN HERE
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
  run: false
  collar: 0.25
  ignore_overlaps: false
```

### misc
- input_list : The file contains the path to wavfiles per each line. Please refer to [data/example.list](data/example.list)
- output_dir : The directory path to store the result as [RTTM](https://github.com/nryant/dscore#rttm) format.
- device : "cpu" for cpu running and "cuda" for gpu running

### audio
- sr : sample rate of audio file (default : 16000)

### vad
- ref_vad : If true, the script will search for the file with different file extension (which will be **ref_suffix**) and use the vad results in that file. (default : false)
- merge_vad : If true, the diarization module will assume that vad segment only contains speech from only **one** speaker. (default : false)
- ref_suffix : The file extension contains vad results. Each line contains start and end time of voice segments. See [data/examples/abjxc.lab](data/examples/abjxc.lab) or [data/examples/akthc.lab](data/examples/akthc.lab). (default : .lab)
- pyannote_token : You need to put token to use pyannote vad model. Please visit [here](https://github.com/pyannote/pyannote-audio#tldr) and [here](https://huggingface.co/settings/tokens) for more information.
- onset, offset, min_duration_on, min_duration_off : Configuration for pyannote vad model. 

### embedding
- win_length : Input length for speaker model (sec) when extracting embeddings with sliding window manner. (default : 1.5)
- hop_length : Hop length for speaker model (sec) when extracting embeddings with sliding window manner. (default : 0.5)

### cluster
- num_cluster : If known, you can put the number of speakers in the wavfile in here. (default : None)
- threshold : threshold for AHC clustering if num_cluster == None (default : 0.8)
- normalize : If true, the embeddings are l2-normalized. (default : false)

### eval
- run : If true, the script will evalute the performance with ground truth. The corresponding rttm file should exist in the same directory (with .rttm file extension) (default: false)
- collar : Duration of collars removed from evaluation around boundaries of groundtruth segments. (default : 0.25)
- ignore_overlaps : If true, we ignore the overlapping segments during evaluation. (default : false)

## Test
- You could run the module with example wavfiles in `data/examples`. The result should be DER : 0.16% and JER : 2.02%.
- The performance on VoxConverse with default configuration is:

|                      | DER   | JER    |
|----------------------|-------|--------|
| VoxConverse dev set  | 5.65% | 17.15% |
| VoxConverse test set | 7.99% | 25.13% |

## Caveat
- Note that this module doesn't consider **overlapping speech** during diarization.

## References
We thank people who kindly open-source the models I used here.

```
@inproceedings{bredin2020pyannote,
  title={Pyannote. audio: neural building blocks for speaker diarization},
  author={Bredin, Herv{\'e} and Yin, Ruiqing and Coria, Juan Manuel and Gelly, Gregory and Korshunov, Pavel and Lavechin, Marvin and Fustes, Diego and Titeux, Hadrien and Bouaziz, Wassim and Gill, Marie-Philippe},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7124--7128},
  year={2020},
  organization={IEEE}
}
```

```
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
