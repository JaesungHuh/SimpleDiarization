from omegaconf import DictConfig, OmegaConf
from diarize.diarize import DiarizationModule
from diarize.score import calculate_score
import hydra
import os
import sys

def read_inputlist(input_list):
    if os.path.isfile(input_list) == False:
        print("No such file : ", input_list)
        sys.exit(1)

    wav_list = []
    with open(input_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if os.path.isfile(line) == False:
                print("No such wav file : ", line)
                continue
            else:
                wav_list.append(line)
    
    print("# of wav files : ", len(wav_list))

    return wav_list


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Read the wavfiles
    input_list = cfg.misc.input_list
    wav_list = read_inputlist(input_list)

    # Initialize diarize module
    diarize_module = DiarizationModule(cfg)
    
    ref_rttm_list = []
    sys_rttm_list = []

    for wav_file in wav_list:
        ref_rttm = wav_file.replace('.wav', '.rttm')
        if cfg.vad.ref_vad == True:
            vad_file = wav_file.replace('.wav', cfg.vad.ref_suffix)
            if os.path.isfile(epd_file) == False:
                raise print("No such file : ", vad_file)
        else:
            vad_file = ""

        sys_rttm = diarize_module.run(wav_file, ref_rttm, vad_file)

        ref_rttm_list.append(ref_rttm)
        sys_rttm_list.append(sys_rttm)
    
    # Evaluation
    if cfg.eval.run:
        der, jer = calculate_score(ref_rttm_list, 
                                            sys_rttm_list, 
                                            p_table=cfg.eval.p_table, 
                                            collar=cfg.eval.collar, 
                                            ignore_overlaps=cfg.eval.ignore_overlaps)
        print("OVERALL DER : {:.02f}% JER : {:.02f}%".format(der, jer))

if __name__ == "__main__":
    main()