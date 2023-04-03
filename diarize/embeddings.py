import sys
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import numpy as np
from collections import OrderedDict

class EmbeddingModule():
    def __init__(self, cfg):
        self.cfg = cfg
        self.merge_vad = cfg.vad.merge_vad
        hyperparameters = {"device": cfg.misc.device}
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts=hyperparameters)

    def extract_embeddings(self, input_file, vad_segments):
        embeddings = OrderedDict()
        signal, fs = torchaudio.load(input_file)
        assert fs == self.cfg.audio.sr

        for (s, e, seg_id) in vad_segments:
            signal_temp = signal[:, int(s * fs): int(e * fs)]
            embedding = self.classifier.encode_batch(signal_temp)
            embedding = embedding.cpu().numpy()
            if seg_id not in embeddings:
                embeddings[seg_id] = []
            embeddings[seg_id].append(np.squeeze(embedding))
        
        if self.cfg.vad.merge_vad:
            # Merge_vad option assuems max. one speaker per vad_segment
            # we average the speaker embeddings in this caase
            final_embeddings = [np.mean(seg, axis=0) for seg_id, seg in embeddings.items()]
        else:
            final_embeddings = []
            for seg_id, seg in embeddings.items():
                final_embeddings = final_embeddings + seg

        final_embeddings = np.stack(final_embeddings, axis=0)
        return final_embeddings
        