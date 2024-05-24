# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT style dataset."""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch_xla
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import MemoryEfficientBlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import deallocate_indexed_dataset_memory
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import make_dataset as make_indexed_dataset
from nemo.core import Dataset
from nemo.utils import logging
from transformers.utils import is_torch_tpu_available
from nemo.collections.common.tokenizers.tokenizer_spec import FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD, EOD

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    import torch_xla.core.xla_model as xm
    ## NEURON ##
    def _create_tensor(t, fn_key):
        _fn_map = {
            "DoubleTensor": torch.DoubleTensor,
            "FloatTensor": torch.FloatTensor,
            "IntTensor": torch.IntTensor,
            "LongTensor": torch.LongTensor
        }
        return _fn_map[fn_key](t).to(xm.xla_device())

    def _create_device():
        return xm.xla_device()

    torch.cuda.DoubleTensor = lambda t: _create_tensor(t, "DoubleTensor")
    torch.cuda.FloatTensor = lambda t: _create_tensor(t, "FloatTensor")
    torch.cuda.IntTensor = lambda t: _create_tensor(t, "IntTensor")
    torch.cuda.LongTensor = lambda t: _create_tensor(t, "LongTensor")
    torch.cuda.current_device = _create_device
except:
    print("Unable to override")

USING_TORCH_VERSION2 = torch.__version__.startswith('2')

def build_dataset(cfg, trainer, data_prefix, data_impl, num_samples, seq_length, seed, skip_warmup, tokenizer, name):
    def _build_dataset(current_data_prefix, current_num_samples):
        indexed_dataset = get_indexed_dataset_(current_data_prefix, data_impl, skip_warmup)
        total_num_of_documents = indexed_dataset.sizes.shape[0]
        # Print stats about the splits.
        logging.info(' > dataset split:')
        logging.info('     Total {} documents is : {} '.format(name, total_num_of_documents))
        drop_last = True
        if name == "valid":
            drop_last = cfg.data.get("validation_drop_last", True)
        dataset = GPTDataset(
            cfg,
            trainer,
            tokenizer,
            name,
            current_data_prefix,
            np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32),
            indexed_dataset,
            current_num_samples,
            seq_length,
            seed,
            drop_last=drop_last,
        )
        return dataset

    if len(data_prefix) == 1:
        return _build_dataset(data_prefix[0], num_samples)

    else:
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, datasets_num_samples = output
        datasets = []
        for i in range(len(prefixes)):
            dataset = _build_dataset(prefixes[i], datasets_num_samples[i])
            datasets.append(dataset)
        return MemoryEfficientBlendableDataset(datasets, weights, num_samples)


def build_train_valid_test_datasets(
    cfg,
    trainer,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    tokenizer,
):
    if isinstance(data_prefix, DictConfig):
        assert (
            data_prefix.get('train') is not None
            and data_prefix.get('test') is not None
            and data_prefix.get('validation') is not None
        ), f"Data prefix dictionary should have train, test and validation keys.  data_prefix currently has only {data_prefix.keys()}"
        if cfg.data.splits_string is not None:
            logging.warning(cfg.data.splits_string + " ignored since data prefix is of type dictionary.")
        train_ds = build_dataset(
            cfg,
            trainer,
            data_prefix["train"],
            data_impl,
            int(train_valid_test_num_samples[0]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "train",
        )
        validation_ds = build_dataset(
            cfg,
            trainer,
            data_prefix["validation"],
            data_impl,
            int(train_valid_test_num_samples[1]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "valid",
        )
        test_ds = build_dataset(
            cfg,
            trainer,
            data_prefix["test"],
            data_impl,
            int(train_valid_test_num_samples[2]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "test",
        )
        return train_ds, validation_ds, test_ds

    else:
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(
                cfg,
                trainer,
                data_prefix[0],
                data_impl,
                splits_string,
                train_valid_test_num_samples,
                seq_length,
                seed,
                skip_warmup,
                tokenizer,
            )

        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                cfg,
                trainer,
                prefixes[i],
                data_impl,
                splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length,
                seed,
                skip_warmup,
                tokenizer,
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        train_n, valid_n, test_n = map(sum, zip(*datasets_train_valid_test_num_samples))

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = MemoryEfficientBlendableDataset(train_datasets, weights, train_n)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = MemoryEfficientBlendableDataset(valid_datasets, weights, valid_n)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = MemoryEfficientBlendableDataset(test_datasets, weights, test_n)

        return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    cfg,
    trainer,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    tokenizer,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    logging.info(' > dataset split:')

    def print_split_stats(name, index):
        logging.info('    {}:'.format(name))
        logging.info(
            '     document indices in [{}, {}) total of {} '
            'documents'.format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32)
            drop_last = True
            if name == "valid":
                drop_last = cfg.data.get("validation_drop_last", True)
            dataset = GPTDataset(
                cfg,
                trainer,
                tokenizer,
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
                drop_last=drop_last,
            )
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    logging.info(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
    logging.info(' > finished creating indexed dataset in {:4f} ' 'seconds'.format(time.time() - start_time))
    logging.info('    number of documents: {}'.format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class GPTDataset(Dataset):
    def __init__(
        self,
        cfg,
        trainer,
        tokenizer,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        drop_last=True,
    ):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        super().__init__()
        self.tokenizer = tokenizer
        self.name = name
        self.indexed_dataset = indexed_dataset
        self.drop_last = drop_last
        self.seq_length = seq_length
        self.np_rng = np.random.RandomState(seed=seed) # rng state for FIM

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        self.reset_position_ids = cfg.data.get('reset_position_ids', False)
        self.reset_attention_mask = cfg.data.get('reset_attention_mask', False)
        self.fim_rate = cfg.data.get('fim_rate', 0)
        self.fim_spm_rate = cfg.data.get('fim_spm_rate', 0)
        self.eod_mask_loss = cfg.data.get('eod_mask_loss', False)
        self.eos_id = tokenizer.eos_id
        self.eod_id = tokenizer.eos_id # as defined in https://github.com/bigcode-project/Megatron-LM/blob/bd0aaba3492b441d7f186bb1159fc21e1dcd7a72/megatron/tokenizer/tokenizer.py#L288
        self.no_seqlen_plus_one_input_tokens = cfg.data.get('no_seqlen_plus_one_input_tokens', False)
        self.add_extra_token = 1
        self.suffix_tok_id, self.prefix_tok_id, self.middle_tok_id, self.pad_tok_id = (self.tokenizer.token_to_id(tok) for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD])

        if self.no_seqlen_plus_one_input_tokens:
            self.add_extra_token = 0

        # save index mappings to a configurable dir
        self.index_mapping_dir = cfg.data.get('index_mapping_dir', None)

        # create index_mapping_dir on rank 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.index_mapping_dir is not None and not os.path.isdir(self.index_mapping_dir):
                    os.makedirs(self.index_mapping_dir)
            torch.distributed.barrier()

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name,
            data_prefix,
            documents,
            self.indexed_dataset.sizes,
            num_samples,
            seq_length,
            seed,
            index_mapping_dir=self.index_mapping_dir,
            drop_last=drop_last,
            add_extra_token=self.add_extra_token,
        )
        deallocate_indexed_dataset_memory(self.indexed_dataset)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def _get_text(self, idx: int) -> np.ndarray:

        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(
                self.doc_idx[doc_index_f], offset=offset_f, length=offset_l - offset_f + self.add_extra_token
            )
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(
                self.indexed_dataset.get(self.doc_idx[doc_index_l], length=offset_l + self.add_extra_token)
            )
            sample = np.concatenate(sample_list)
        ###
        # Code from: https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L109
        # TODO(Hailey): can merge the code below this line with code above this line.
        # TODO(Hailey), cont: above already iterates through loop, so just add the permuting in there?
        sample = np.array(sample, dtype=np.int64)
        sample_len = sample.shape[0]
        # # print(sample, sample.shape)
        # # do FIM here, if enabled
        # TODO: Do we handle the following point from FIM paper?
        # To transform data in the character space for context-level FIM, the tokenized documents have to be decoded back into strings before FIM augmentation. Depending on the vocabulary, some care has to be given to ensure decoding does not introduce any spurious characters into training. For example, utf-8 characters are encoded as multiple tokens with a BPE vocabulary; they can result in fragments from chunking and fail to decode. To prevent unforeseen errors midway through training, we encourage checking for these fragments at the beginning or end of a context and removing them.
        fim_rate = self.fim_rate

        if fim_rate != 0:
            assert (fim_rate <= 1 and fim_rate >= 0), "FIM rate must be a probability 0 <= rate <= 1"

            eod = self.eod_id
            segment_breaks = np.argwhere(sample == eod) # split sample by document

            if segment_breaks.shape != (0, 1): # then there is an EOD token in this example
                curr_start_position = 0
                new_samples = []
                for loc in np.nditer(segment_breaks):
                    # Only permute non-empty segments.
                    if loc - curr_start_position > 0:
                        # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                        permuted, self.np_rng = \
                            permute(sample[curr_start_position:loc], self.np_rng, self.fim_rate, self.fim_spm_rate, self.tokenizer, truncate_or_pad=False,
                                    suffix_tok_id=self.suffix_tok_id, prefix_tok_id=self.prefix_tok_id, middle_tok_id=self.middle_tok_id, pad_tok_id=self.pad_tok_id)
                        new_samples += [permuted, [eod]]

                    curr_start_position = loc + 1 # jump over the EOD token
                # Permute the segment after the last EOD
                permuted, self.np_rng = \
                    permute(sample[curr_start_position:], self.np_rng, self.fim_rate, self.fim_spm_rate, self.tokenizer, truncate_or_pad=False,
                            suffix_tok_id=self.suffix_tok_id, prefix_tok_id=self.prefix_tok_id, middle_tok_id=self.middle_tok_id, pad_tok_id=self.pad_tok_id)
                new_samples.append(permuted)

                sample = np.concatenate(new_samples)
            else:
                sample, self.np_rng = permute(sample, self.np_rng, self.fim_rate, self.fim_spm_rate, self.tokenizer, truncate_or_pad=False,
                                              suffix_tok_id=self.suffix_tok_id, prefix_tok_id=self.prefix_tok_id, middle_tok_id=self.middle_tok_id, pad_tok_id=self.pad_tok_id)

        # Truncate or pad sequence to max-length
        diff = sample.shape[0] - (self.seq_length + self.add_extra_token)
        if diff > 0: # too long
            sample = sample[:(self.seq_length + self.add_extra_token)]
        elif diff < 0: # too short
        #    sample = np.concatenate([sample, np.full((-1 * diff), self.pad_tok_id)])
            logging.info(
                F' > WARNING: Got sample of length: {len(sample)} for sequence length={self.seq_length+self.add_extra_token}, padding the sample to match sequence length'
            )
            sample = np.array(sample, dtype=np.int64)
            sample = np.pad(
                sample, (0, self.seq_length + self.add_extra_token - len(sample)), mode='constant', constant_values=-1
            )

        assert sample.shape[0] == (self.seq_length + self.add_extra_token)
        # end FIM-specific code
        ###
        #if len(sample) != (self.seq_length + self.add_extra_token):
        #    logging.info(
        #        F' > WARNING: Got sample of length: {len(sample)} for sequence length={self.seq_length+self.add_extra_token}, padding the sample to match sequence length'
        #    )
        #    sample = np.array(sample, dtype=np.int64)
        #    sample = np.pad(
        #        sample, (0, self.seq_length + self.add_extra_token - len(sample)), mode='constant', constant_values=-1
        #    )
        return sample.astype(np.int64)

    def __getitem__(self, idx):
        text = torch.from_numpy(self._get_text(idx))
        if self.add_extra_token:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = -1
        attention_mask, loss_mask, position_ids = _create_ltor_masks_and_position_ids(
            tokens, self.eos_id, self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss,
        )
        loss_mask[labels == -1] = 0.0
        tokens[tokens == -1] = 0
        labels[labels == -1] = 0
        # Negative index comes when we pad the last batch in MegatronPretrainingBatchSampler
        # We make the loss_mask zero to mask out loss from these samples
        if idx == -1:
            logging.info('WARNING: Got -1 as item index. Masking loss from this sample')
            loss_mask = torch.zeros_like(loss_mask)
        return {
            'tokens': tokens,
            'labels': labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
        }


@torch.no_grad()
def _create_ltor_masks_and_position_ids(
    tokens: torch.Tensor, eod_token: int, reset_position_ids: bool, reset_attention_mask: bool, eod_mask_loss: bool,
):
    """Create `attention_mask`, `loss_mask`, and `position_ids`.

    This function is modified :func:`get_ltor_masks_and_position_ids` in nemo/collections/nlp/modules/common/megatron/utils.py:
    `get_ltor_masks_and_position_ids` assumes a microbatch of ``tokens``, i.e. 2D tensor while
    this function assumes ``tokens`` to be 1D tensor.

    Args:
        tokens: A 1D tensor that holds the indices of tokens.
        eod_token:
        reset_position_ids:
        reset_attention_mask:
        eod_mask_loss

    """
    assert tokens.ndim == 1
    seq_length = tokens.numel()
    loss_mask = torch.ones(seq_length, dtype=torch.float)
    if eod_mask_loss:
        loss_mask[tokens == eod_token] = 0.0

    position_ids = torch.arange(seq_length, dtype=torch.int64)
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids:
        # Find indices where EOD token is.
        eod_index = position_ids[tokens == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1
    if is_torch_tpu_available():
        assert (not reset_attention_mask),"Neuron flow does not support attention mask reset"
        attention_mask = torch.tensor([True])
        # Needs to be a dummy tensor since a whole bunch of batch things are done under the hood. 
        # Size does not matter, it will be replaced by device tensor
    else:
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0)
        if reset_attention_mask:
            # Find indices where EOD token is.
            eod_index = position_ids[tokens == eod_token]
            prev_index = 0
            for j in range(eod_index.numel()):
                i = eod_index[j]
                if reset_attention_mask:
                    attention_mask[0, (i + 1) :, : (i + 1)] = 0
        # Convert attention mask to binary.
        attention_mask = attention_mask < 0.5
    return attention_mask, loss_mask, position_ids


def _build_index_mappings(
    name,
    data_prefix,
    documents,
    sizes,
    num_samples,
    seq_length,
    seed,
    index_mapping_dir: str = None,
    drop_last: bool = True,
    add_extra_token: int = 1
):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples, add_extra_token)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    if index_mapping_dir is not None:
        _filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
    else:
        _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}sl'.format(seq_length)
    _filename += '_{}s'.format(seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0:
        if (
            (not os.path.isfile(doc_idx_filename))
            or (not os.path.isfile(sample_idx_filename))
            or (not os.path.isfile(shuffle_idx_filename))
        ):

            logging.info(' > WARNING: could not find index map files, building ' 'the indices on rank 0 ...')

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                print(' > only one epoch required, setting ' 'separate_last_epoch to False', flush=True)

            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - add_extra_token
                ) // seq_length
                last_epoch_num_samples = num_samples - num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, 'last epoch number of samples should be non-negative.'
                num_samples_per_epoch = (tokens_per_epoch - add_extra_token) // seq_length
                assert last_epoch_num_samples < (
                    num_samples_per_epoch + 1
                ), 'last epoch number of samples exceeded max value.'
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = last_epoch_num_samples < int(0.80 * num_samples_per_epoch)
                if separate_last_epoch:
                    string = (
                        ' > last epoch number of samples ({}) is smaller '
                        'than 80% of number of samples per epoch ({}), '
                        'setting separate_last_epoch to True'
                    )
                else:
                    string = (
                        ' > last epoch number of samples ({}) is larger '
                        'than 80% of number of samples per epoch ({}), '
                        'setting separate_last_epoch to False'
                    )
                print(string.format(last_epoch_num_samples, num_samples_per_epoch), flush=True)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            logging.info(
                ' > elasped time to build and save doc-idx mapping '
                '(seconds): {:4f}'.format(time.time() - start_time)
            )
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            try:
                from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import compile_helper

                compile_helper()
                from nemo.collections.nlp.data.language_modeling.megatron import helpers
            except ImportError:
                raise ImportError(
                    f'Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file.'
                )

            sample_idx = helpers.build_sample_idx(
                sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch, drop_last, add_extra_token
            )
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                              num_epochs, tokens_per_epoch, drop_last, add_extra_token)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            logging.info(
                ' > elasped time to build and save sample-idx mapping '
                '(seconds): {:4f}'.format(time.time() - start_time)
            )
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_, sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            logging.info(
                ' > elasped time to build and save shuffle-idx mapping'
                ' (seconds): {:4f}'.format(time.time() - start_time)
            )

    # torch.distributed.barrier()
    import torch_xla.core.xla_model as xm
    xm.rendezvous('shuffle_idx_mapping')
    #counts = torch.cuda.LongTensor([1])
    #torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group())
    #torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())

    # assert counts[0].item() == (
    #     torch.distributed.get_world_size()
    #     // torch.distributed.get_world_size(group=parallel_state.get_tensor_model_parallel_group())
    # )

    # Load mappings.
    start_time = time.time()
    logging.info(' > loading doc-idx mapping from {}'.format(doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    logging.info(' > loading sample-idx mapping from {}'.format(sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    logging.info(' > loading shuffle-idx mapping from {}'.format(shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    logging.info('    loaded indexed file in {:3.3f} seconds'.format(time.time() - start_time))
    logging.info('    total number of samples: {}'.format(sample_idx.shape[0]))
    logging.info('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples, add_extra_token=1):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - add_extra_token) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs - 1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch, drop_last=True, add_extra_token=1):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    if not drop_last:
        num_samples = -(-(num_epochs * tokens_per_epoch - add_extra_token) // seq_length)
    else:
        num_samples = (num_epochs * tokens_per_epoch - add_extra_token) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + add_extra_token
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += remaining_seq_length + doc_length - add_extra_token
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                if doc_idx_index == (len(doc_idx) - 1):
                    assert (
                        sample_index == num_samples
                    ), F"sample_index={sample_index} and num_samples={num_samples} should be the same"
                    doc_offset = sizes[doc_idx[doc_idx_index]] - add_extra_token
                    break
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print(
        ' > building shuffle index with split [0, {}) and [{}, {}) '
        '...'.format(num_samples, num_samples, total_size),
        flush=True,
    )

    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))

# From https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L339
def permute(sample, np_rng, fim_rate, fim_spm_rate, tokenizer, truncate_or_pad=True,
            suffix_tok_id=None, prefix_tok_id=None, middle_tok_id=None, pad_tok_id=None):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """

    if np_rng.binomial(1, fim_rate): # sample bernoulli dist

        contents = tokenizer.ids_to_text(sample)

        try:
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(np_rng.randint(low=0, high=len(contents) + 1, size=2))
            boundaries.sort()
        except ValueError as e:
            print(len(contents), contents)
            print(e)
            raise e

        prefix = contents[:boundaries[0]]
        middle = contents[boundaries[0]:boundaries[1]]
        suffix = contents[boundaries[1]:]

        prefix = np.array([*tokenizer.text_to_ids(prefix)], dtype=np.int64)
        middle = np.array([*tokenizer.text_to_ids(middle)], dtype=np.int64)
        suffix = np.array([*tokenizer.text_to_ids(suffix)], dtype=np.int64)

        # here we truncate each given segment to fit the same length as it was before
        # A consequence is that we never reach the end of a file?
        # we should rather truncate at the context-level
        if truncate_or_pad:
            # need to make same length as the input. Take the 3 sentinel tokens into account
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - sample.shape[0]
            if diff > 0: # too long
                if suffix.shape[0] <= diff: # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
                    return sample, np_rng
                suffix = suffix[:suffix.shape[0] - diff]
            elif diff < 0: # too short
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate([
                [prefix_tok_id, suffix_tok_id], suffix,
                [middle_tok_id], prefix, middle
            ])
        else:
            # PSM
            new_sample = np.concatenate([
                [prefix_tok_id], prefix,
                [suffix_tok_id], suffix,
                [middle_tok_id], middle
            ])

    else:
        # don't do FIM preproc
        new_sample = sample

    return new_sample, np_rng