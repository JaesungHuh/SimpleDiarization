import hydra
import torchaudio

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

class VADModule():
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.vad.method == 'pyannote':
            print("Initializing VADModule : pyannote segmentation")
            self.token = cfg.vad.token
            self.model = Model.from_pretrained("pyannote/segmentation", use_auth_token=self.token)
            hyperparameters = {"onset": cfg.vad.onset, 
                                "offset": cfg.vad.offset,
                                "min_duration_on": cfg.vad.min_duration_on,
                                "min_duration_off": cfg.vad.min_duration_off}
            self.pipeline = VoiceActivityDetection(segmentation=self.model)
            self.pipeline.instantiate(hyperparameters)
            print("Finished VADModule : pyannote segmentation")
        else:
            print("No such VAD method : ", cfg.vad.method)
            sys.exit(1)
    
    def get_pyannote_segments(self, input_file):
        vad_segments = []
        vad = self.pipeline(input_file)
        for x, turn in vad.itertracks():
            vad_segments.append((x.start, x.end))
        starts, ends = self.sliding_window(vad_segments)

        return starts, ends
        
    def sliding_window(self, vad_segments):
        starts, ends = [], []
        
        # win_length and hop_length should be second
        chunk_len = self.cfg.embedding.win_length
        chunk_overlap = self.cfg.embedding.hop_length

        for seg in vad_segments:
            ii = 0
            start = seg[0]
            end   = seg[1]

            cur_start = round(start, 3)

            while cur_start + chunk_len < end:
                starts.append(round(cur_start, 3))
                ends.append(round(cur_start + chunk_len, 3))
                cur_start += chunk_overlap
                ii += 1
            if cur_start < end:
                starts.append(round(cur_start, 3))
                ends.append(round(end, 3))
                ii += 1
        
        return starts, ends

if __name__ == '__main__':
    main()