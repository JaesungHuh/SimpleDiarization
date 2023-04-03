import os
import sys

cur = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(cur, '../VoxSRC2022')))

from compute_diarisation_metrics import *

def calculate_score(ref_rttm, sys_rttm, p_table=False, collar=0.01, ignore_overlaps=False):
    ref_turns, _ = load_rttms(ref_rttm)
    sys_turns, _ = load_rttms(sys_rttm)

    uem = gen_uem(ref_turns, sys_turns)
    ref_turns = trim_turns(ref_turns, uem)
    sys_turns = trim_turns(sys_turns, uem)

    ref_turns = merge_turns(ref_turns)
    sys_turns = merge_turns(sys_turns)

    check_for_empty_files(ref_turns, sys_turns, uem)

    file_score, global_score = score(ref_turns, sys_turns, uem, step=0.01, jer_min_ref_dur = 0.0, collar=collar, ignore_overlaps=ignore_overlaps)
    der     = global_score.der
    jer     = global_score.jer

    return der, jer