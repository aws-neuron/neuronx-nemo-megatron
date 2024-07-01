import argparse
import os
import pathlib
import torch

def merge_mlp_weights(args):
    input_path = args.input_path
    output_path = args.output_path
    num_layers = args.num_layers
    for i in os.listdir(input_path):
        chkpt_file=pathlib.Path(input_path).joinpath(f'{i}/model_optim_rng.ckpt')
        print(chkpt_file)
        chkpt = torch.load(chkpt_file)
        for j in range(0, num_layers):
            chkpt['state_dict'][f'model.language_model.encoder.layers.{j}.mlp.dense_h_to_4h.weight'] = \
            torch.concat([chkpt['state_dict'][f'model.language_model.encoder.layers.{j}.mlp.dense_h_to_4h.weight'], 
                          chkpt['state_dict'][f'model.language_model.encoder.layers.{j}.mlp.dense_h_to_4h_2.weight']], dim=0)
            chkpt['state_dict'].pop(f'model.language_model.encoder.layers.{j}.mlp.dense_h_to_4h_2.weight')
        new_chkpt_dir=pathlib.Path(output_path).joinpath(f'{i}')
        new_chkpt_dir.mkdir(parents=True, exist_ok=True)
        new_chkpt_file=new_chkpt_dir.joinpath('model_optim_rng.ckpt')
        torch.save(chkpt, new_chkpt_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input checkpoint path",
        required=True
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output checkpoint path",
        required=True
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        help="number of layers in the sharded model checkpoint",
        required=True
    )
    args = parser.parse_args()
    merge_mlp_weights(args)

