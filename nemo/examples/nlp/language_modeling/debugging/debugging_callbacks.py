import pytorch_lightning as pl
import numpy as np
count = 0

class PrintGradientsCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        pass

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        global count
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        num_layers = 1  # total layers divided by pp
        pp = 4
        if True:
            os.makedirs(f'dump/Rank:{xm.get_ordinal()}_{num_layers}_layer', exist_ok=True)
            if xm.get_ordinal()<32: #embeddings on first tp group
                np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/wte_grads_count_{count}.npy", pl_module.model.language_model.embedding.word_embeddings.weight.grad.detach().cpu().numpy())
                np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/wte_weights_count_{count}.npy", pl_module.model.language_model.embedding.word_embeddings.weight.detach().cpu().numpy())

            if -1 < (xm.get_ordinal() - (32 * (pp - 1))) < 32: #embeddings on last tp group
                np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/wte_grads_count_{count}.npy", pl_module.model.word_embeddings.weight.grad.detach().cpu().numpy())
                np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/wte_weights_count_{count}.npy", pl_module.model.word_embeddings.weight.detach().cpu().numpy())

            for i in range(num_layers):
                # Access each ParallelTransformerLayer
                layer = pl_module.model.language_model.encoder.layers[i]

                pre_norm = layer.input_layernorm
                weights = pre_norm.weight.detach().cpu().numpy()
                print(weights.shape)
                np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/layer_{i}_input_layernorm_weights_count_{count}.npy", weights)
                print(f'Saved weights of layer_{i}_input_layernorm at count {count}.')

                if pre_norm.weight.grad is not None:
                    gradients = pre_norm.weight.grad.detach().cpu().numpy()
                    np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/layer_{i}_pre_norm_gradients_count_{count}.npy", gradients)
                    print(f'Saved gradients of layer_{i}_pre_norm at count {count}.')
                else:
                    print(f'No gradients available for layer_{i}_pre_norm.')

                # Save weights and gradients for self_attention layers
                self_attention_layer = layer.self_attention
                for att_layer in [self_attention_layer.query_key_value, self_attention_layer.dense]:
                    weights = att_layer.weight.detach().cpu().numpy()
                    print(weights.shape)
                    np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/layer_{i}_self_attention_{att_layer}_weights_count_{count}.npy", weights)
                    print(f'Saved weights of layer_{i}_self_attention_{att_layer} at count {count}.')

                    if att_layer.weight.grad is not None:
                        gradients = att_layer.weight.grad.detach().cpu().numpy()
                        np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/layer_{i}_self_attention_{att_layer}_gradients_count_{count}.npy", gradients)
                        print(f'Saved gradients of layer_{i}_self_attention_{att_layer} at count {count}.')
                    else:
                        print(f'No gradients available for layer_{i}_self_attention_{att_layer}.')

                # Save weights and gradients for mlp layers
                mlp_layer = layer.mlp
                for mlp in [mlp_layer.dense_h_to_4h, mlp_layer.dense_4h_to_h]:
                    weights = mlp.weight.detach().cpu().numpy()
                    print(weights.shape)

                    np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/layer_{i}_mlp_{mlp}_weights_count_{count}.npy", weights)
                    print(f'Saved weights of layer_{i}_mlp_{mlp} at count {count}.')

                    if mlp.weight.grad is not None:
                        gradients = mlp.weight.grad.detach().cpu().numpy()
                        np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/layer_{i}_mlp_{mlp}_gradients_count_{count}.npy", gradients)
                        print(f'Saved gradients of layer_{i}_mlp_{mlp} at count {count}.')
                    else:
                        print(f'No gradients available for layer_{i}_mlp_{mlp}.')

                post_norm = layer.post_attention_layernorm
                weights = post_norm.weight.detach().cpu().numpy()
                np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/layer_{i}_post_attention_layernorm_weights_count_{count}.npy", weights)
                print(f'Saved weights of layer_{i}_post_attention_layernorm at count {count}.')

                if post_norm.weight.grad is not None:
                    gradients = post_norm.weight.grad.detach().cpu().numpy()
                    np.save(f"dump/Rank:{xm.get_ordinal()}_{num_layers}_layer/layer_{i}_post_norm_gradients_count_{count}.npy", gradients)
                    print(f'Saved gradients of layer_{i}_post_norm at count {count}.')
                else:
                    print(f'No gradients available for layer_{i}_post_norm.')
            count += 1
        xm.rendezvous("Saving weights")
        pass