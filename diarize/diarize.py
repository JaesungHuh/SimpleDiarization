import os, sys
cur = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(cur))
from utils import write_rttm
from vad import VADModule
from embeddings import EmbeddingModule
from cluster import ClusterModule
from score import calculate_score
from utils import write_rttm, read_vadfile

class DiarizationModule():
    def __init__(self, cfg):
        self.cfg = cfg

        self.vad_module = VADModule(cfg)
        self.embedding_module = EmbeddingModule(cfg)
        self.cluster_module = ClusterModule(cfg)

        # Create output_dir
        self.output_dir = self.cfg.misc.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, wav_file, ref_rttm_file, vad_file = ""):
        # Write sys_rttm to {output_dir}/{ref_rttm_file}
        # dataname = os.path.basename(wav_file).replace('.wav', '')

        # VAD
        if vad_file == "" or (os.path.isfile(vad_file) == False):
            starts, ends = self.vad_module.get_pyannote_segments(wav_file)
        else:
            starts, ends = read_vadfile(vad_file)
        vad_segments = list(zip(starts, ends))

        # Extract embeddings
        embeddings = self.embedding_module.extract_embeddings(wav_file, vad_segments)

        # Clustering
        SEL_tuples = self.cluster_module.cluster(embeddings, starts, ends)

        # Write the resultant rttm file
        sys_rttm   = ref_rttm_file.split('/')[-1]
        sys_rttm   = os.path.join(self.output_dir, sys_rttm)
        write_rttm(SEL_tuples, sys_rttm)
        return sys_rttm
        
    def calcDERofRTTMfiles(self, ref_rttm_files, sys_rttm_files):
        """ calculate DER between `ref_rttm_files` and `sys_rttm_files`. each argument contains list of RTTM file names.

        Parameters
        ----------
        ref_rttm_files
        sys_rttm_files

        Returns
        -------

        """
        der, jer = calculate_score(ref_rttm_files, sys_rttm_files, p_table=self.p_table, collar=self.collar, ignore_overlaps=self.ignore_overlaps)

        return der, jer
