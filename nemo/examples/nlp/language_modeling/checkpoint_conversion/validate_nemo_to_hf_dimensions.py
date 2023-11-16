import torch
import argparse

def check_dimensions(args):
    """
    Compares dimensions of all layers in converted model to
    provided MStar model.
    """
    mstar_model = torch.load(args.path_to_mstar)
    converted_model = torch.load(args.path_to_nemo_converted)

    for layer_name, tensor in converted_model.items():
        m_shape = mstar_model[layer_name].shape
        n_shape = converted_model[layer_name].shape
        assert m_shape == n_shape, f"Shapes for {layer_name} do not match: MStar: {m_shape}, Converted Nemo: {n_shape}"
    print(f"Dimensions of all {len(converted_model)} tensors match.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_mstar",
        type=str,
        help="Path to the original mstar model",
    )
    parser.add_argument(
		"--path_to_nemo_converted",
		type=str,
		help="Path to the output of `convert_nemo_checkpoint_to_hf.py`",
		required=True
	)
    args = parser.parse_args()
    check_dimensions(args)
