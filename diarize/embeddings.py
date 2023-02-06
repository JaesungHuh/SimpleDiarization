import sys
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import numpy as np

class EmbeddingModule():
    def __init__(self, cfg):
        print("Initializing Embedding extractor")
        self.cfg = cfg
        hyperparameters = {"device": cfg.misc.device}
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts=hyperparameters)
        print("Finished initializing Embedding extractor")

    def extract_embeddings(self, input_file, vad_segments):
        embeddings = []
        signal, fs = torchaudio.load(input_file)
        assert fs == self.cfg.audio.sr
        for (s, e) in vad_segments:
            signal_temp = signal[:, int(s * fs): int(e * fs)]
            embedding = self.classifier.encode_batch(signal_temp)
            embedding = embedding.cpu().numpy()
            embeddings.append(np.squeeze(embedding))
        
        return np.stack(embeddings, axis=0)
        