import os
import argparse
import yaml
import tqdm
import time

from diarize.diarize import DiarizationModule
from diarize.score import calculate_score
from diarize.utils import Dict2ObjParser, read_inputlist


def parse_args():
    """Argparse arguments

    Returns:
        args: contains the path to cfg_file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='conf/config.yaml')
    args = parser.parse_args()

    return args


def main():
    """main function for speaker diarization"""

    args = parse_args()

    # Read the yaml file
    with open(args.cfg_file, 'r') as f_cfg:
        nested_dict = yaml.safe_load(f_cfg)
        cfg = Dict2ObjParser(nested_dict).parse()

    # Read the wavfiles
    input_list = cfg.misc.input_list
    wav_list = read_inputlist(input_list)

    # Initialize diarize module
    diarize_module = DiarizationModule(cfg)

    # Diarize each wavfiles
    ref_rttm_list, sys_rttm_list = [], []
    for wav_file in tqdm.tqdm(wav_list):
        start_time = time.time()
        ref_rttm = wav_file.replace('.wav', '.rttm')
        if cfg.vad.ref_vad == True:
            vad_file = wav_file.replace('.wav', cfg.vad.ref_suffix)
            if not os.path.isfile(vad_file):
                raise ValueError("No such file : ", vad_file)
        else:
            vad_file = ""

        sys_rttm = diarize_module.run(wav_file, ref_rttm, vad_file)
        total_time += end_time - start_time
        ref_rttm_list.append(ref_rttm)
        sys_rttm_list.append(sys_rttm)

    # Evaluation
    if cfg.eval.run:
        der, jer = calculate_score(ref_rttm_list,
                                   sys_rttm_list,
                                   collar=cfg.eval.collar,
                                   ignore_overlaps=cfg.eval.ignore_overlaps)
        print(f"OVERALL DER : {der:.02f}% JER : {jer:.02f}%")


if __name__ == "__main__":
    main()
