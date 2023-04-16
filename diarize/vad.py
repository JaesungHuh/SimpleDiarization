from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection


class VADModule():
    def __init__(self, cfg):
        self.cfg = cfg
        if (cfg.vad.ref_vad == False):
            print("Initializing VADModule : pyannote segmentation")
            self.token = cfg.vad.pyannote_token
            self.model = Model.from_pretrained("pyannote/segmentation", use_auth_token=self.token)
            hyperparameters = {"onset": cfg.vad.onset, 
                                "offset": cfg.vad.offset,
                                "min_duration_on": cfg.vad.min_duration_on,
                                "min_duration_off": cfg.vad.min_duration_off}
            self.pipeline = VoiceActivityDetection(segmentation=self.model)
            self.pipeline.instantiate(hyperparameters)
        else:
            print("Initialize dummy object for reading the vad file")
    
    def get_pyannote_segments(self, input_file):
        # Vad using pyannote segmentation model
        vad_segments = []
        vad = self.pipeline(input_file)
        for x, turn in vad.itertracks():
            vad_segments.append([x.start, x.end])

        return vad_segments
        
    def sliding_window(self, vad_segments):
        # Chop the vad segments with sliding window with win_length and hop_length
        starts, ends, seg_ids = [], [], []
        
        # win_length and hop_length should be second
        chunk_len = self.cfg.embedding.win_length
        chunk_overlap = self.cfg.embedding.hop_length

        for seg_id, seg in enumerate(vad_segments):
            start = seg[0]
            end   = seg[1]

            cur_start = round(start, 3)

            while cur_start + chunk_len < end:
                starts.append(round(cur_start, 3))
                ends.append(round(cur_start + chunk_len, 3))
                cur_start += chunk_overlap
                seg_ids.append(seg_id)
            if cur_start < end:
                starts.append(round(cur_start, 3))
                ends.append(round(end, 3))
                seg_ids.append(seg_id)
        
        return starts, ends, seg_ids
    
    def merge_intervals(self, intervals):
        # Merge all overlapping intervals and return an array of the non-overlapping intervals
        # Originally from https://leetcode.com/problems/merge-intervals/solutions/3129905/merge-intervals-by-using-stack-simple-explaination/?languageTags=python3
        intervals.sort(key=lambda x:x[0])
        stack = []
        for i in range(0,len(intervals)):
            if stack and stack[0][1] >= intervals[i][0]:
                #Overlapping condition.... Update the end point accordingly...
                stack[0][1] = max(stack[0][1],intervals[i][1])
            else:
                stack.insert(0,intervals[i])
        stack.sort(key=lambda x:x[0])
        return stack