import os
import re
import gc
import argparse
from pathlib import Path, PurePath
from os.path import join
from glob import glob

import torch
import torch_xla.utils.serialization as xser
import nemo.collections.nlp.parts.serialization as nser

def get_tp_pp_degree(path_to_checkpoints):
    dir_name = PurePath(path_to_checkpoints)
    TP = 1
    PP = 1

    for folder in os.listdir(dir_name):
        mp_search = re.search('mp_rank_[\d]*', folder)
        if mp_search:
            TP = max(TP, 1+int(mp_search[0].split('mp_rank_')[1]))
    for folder in os.listdir(dir_name):
        tp_search = re.search('tp_rank_[\d]*', folder)
        if tp_search:
            TP = max(TP, 1+int(tp_search[0].split('tp_rank_')[1]))

        pp_search = re.search('pp_rank_[\d]*', folder)
        if pp_search:
            PP = max(PP, 1+int(pp_search[0].split('pp_rank_')[1]))
    
    return TP, PP

def convert_checkpoint(path_to_checkpoints, output_path, ckpt_name):
    TP, PP = get_tp_pp_degree(path_to_checkpoints)
    os.makedirs(output_path, exist_ok=False)
    if PP == 1 and TP == 1:
        ckpt_str = str()
    else:
        ckpt_str = f"tp_rank_*_pp_rank_*" if PP > 1 else "mp_rank_*"

    template = join(path_to_checkpoints, ckpt_str, ckpt_name)

    paths = sorted(glob(template))
    for i in paths:
        print(f"Converting path {i}")
        ckpt_str = os.path.basename(os.path.dirname(i))
        os.mkdir(join(output_path, ckpt_str))
        loaded = xser.load(i)
        nser.save(loaded, join(output_path, ckpt_str, ckpt_name))
        del loaded
        gc.collect()
    print("Format conversion: SUCCESS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_checkpoints",
        type=str,
        help="Path to the checkpoints",
        required=True
    )
    parser.add_argument(
        "--output_path",
        default="",
        type=str,
        help="Output path",
    )
    parser.add_argument(
        "--checkpoint_name",
        default="",
        type=str,
        help="Checkpoint name",
    )
    args = parser.parse_args()
    convert_checkpoint(args.path_to_checkpoints, args.output_path, args.checkpoint_name)
