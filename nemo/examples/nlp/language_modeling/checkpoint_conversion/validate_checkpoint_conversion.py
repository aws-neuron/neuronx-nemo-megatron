"""
Test case for verifying that the 1.1B model checkpoint loaded by NeMo produces the same result as MStar
Download sample data using:
`aws s3 cp s3://kaena-tempdata/rhsoln/sunda/bedrock/sample_data_for_chkpt_val/ ~/np_data/`
"""
import os
import numpy as np
import torch
import torch.distributed as dist
from apex.transformer import parallel_state
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import GPTModel
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import (
    _create_ltor_masks_and_position_ids,
)


def main():
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size_=1, pipeline_model_parallel_size_=1
    )

    # This config is from 1.1B model
    padded_vocab_size = 34176
    hidden_size = 1920
    max_position_embeddings = 2048
    num_layers = 24
    num_attention_heads = 16
    apply_query_key_layer_scaling = True
    kv_channels = None
    ffn_hidden_size = 7680
    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()
    init_method_std = 0.02
    use_scaled_init_method = True
    hidden_dropout = 0.0
    attention_dropout = 0.0
    ffn_dropout = 0.0
    precision = 32
    normalization = "layernorm"

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.model = GPTModel(
                vocab_size=padded_vocab_size,
                hidden_size=hidden_size,
                max_position_embeddings=max_position_embeddings,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                ffn_hidden_size=ffn_hidden_size,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=init_method_std,
                use_scaled_init_method=use_scaled_init_method,
                fp16_lm_cross_entropy=False,
                use_cpu_initialization=False,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                precision=precision,
                fp32_residual_connection=False,
                activations_checkpoint_granularity=None,
                activations_checkpoint_method=None,
                activations_checkpoint_num_layers=1,
                activations_checkpoint_layers_per_pipeline=None,
                normalization=normalization,
                layernorm_epsilon=1e-5,
                onnx_safe=False,
                bias=True,
                bias_activation_fusion=True,
                bias_dropout_add_fusion=True,
                activation="gelu",
                headscale=False,
                transformer_block_type="pre_ln",
                openai_gelu=False,
                normalize_attention_scores=True,
                position_embedding_type="learned_absolute",
                rotary_percentage=1.0,
                share_embeddings_and_output_weights=False,
                attention_type="multihead",
                masked_softmax_fusion=False,
                gradient_accumulation_fusion=False,
                persist_layer_norm=False,
                sequence_parallel=False,
                transformer_engine=False,
                fp8=False,
                fp8_e4m3=False,
                fp8_hybrid=False,
                fp8_margin=0,
                fp8_interval=1,
                fp8_amax_history_len=1,
                fp8_amax_compute_algo="most_recent",
                reduce_amax=True,
                use_emha=False,
            )

        def load_state_dict(self, checkpoint):
            with torch.no_grad():
                self.model.language_model.output_layer.weight.copy_(
                    checkpoint["language_model.final_layer.weight"]
                )
            del checkpoint["language_model.final_layer.weight"]
            self.model.load_state_dict(checkpoint)

        def forward(
            self, input_ids=None, position_ids=None, labels=None, attention_mask=None
        ):
            return self.model(input_ids, position_ids, attention_mask)

    model = Model()
    # Provide your checkpoint path
    model_chpt = torch.load(
        "/home/ubuntu/rhsoln/nemo_checkpoint/mp_rank_00/megatron_gpt--val_loss=0.00-step=4-consumed_samples=0-last.ckpt"
    )
    model.load_state_dict(model_chpt["state_dict"])

    model.to("cuda")
    data_path = "~/np_data/"
    for file_path in os.listdir(data_path):
        with open(data_path + file_path, "rb") as f:
            np_data = np.load(f, allow_pickle=True)

        inputs = {}
        inputs["input_ids"] = torch.tensor(np_data.item()["input_ids"]).to("cuda")
        attention_mask, loss_mask, position_ids = _create_ltor_masks_and_position_ids(
            inputs["input_ids"][0], -1, False, False, False
        )
        inputs["attention_mask"] = attention_mask.to(torch.bool).to("cuda").unsqueeze(0)
        inputs["position_ids"] = position_ids.to("cuda").unsqueeze(0)
        outputs = model(**inputs)
        assert outputs.shape == np_data.item()["outputs"].shape
        try:
            assert np.allclose(
                outputs.detach().cpu().numpy(),
                np_data.item()["outputs"],
                rtol=1e-02,
                atol=1e-01,
                equal_nan=False,
            )
        except:
            print(f"Failed for {file_path}")


if __name__ == "__main__":
    main()
