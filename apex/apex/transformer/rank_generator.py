from typing import List

def generate_masked_orthogonal_rank_groups(
    world_size: int,
    parallel_size: List[int],
    mask: List[bool],
) -> List[List[int]]:
    """Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example,
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)
        Each DP group combines TP and PP ranks to form a set of GPUs that process different data batches
        but have the same TP and PP configuration.

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example, if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)
        2   2   2   --> Size
        1   2   4   --> Radix
        TP  PP  DP
        0   0   0   --> 0
        0   0   1   --> 4

        1   0   0   --> 1
        1   0   1   --> 5

        0   1   0   --> 2
        0   1   1   --> 6

        1   1   0   --> 3
        1   1   1   --> 7

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).

        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]

        2   3   4   --> Size
        1   2   6   --> Radix
        TP  DP  PP
        0   0   0   --> 0
        1   0   0   --> 1
        0   1   0   --> 2
        1   1   0   --> 3
        0   2   0   --> 4
        1   2   0   --> 5

        0   0   1   --> 6

        1   0   0   --> 1
        1   1   0   --> 3
        1   2   0   --> 5
        ...

        1   0   3   --> 19
        1   1   3   --> 21
        1   2   3   --> 23
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        """
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


class RankGenerator(object):
    def __init__(self, tp: int, dp: int, pp: int, cp: int) -> None:
        """_summary_

        Args:
            tp (int): _description_
            dp (int): _description_
            pp (int): _description_
            cp (int): _description_
            order (str): E.g. cp-tp-dp-pp

        Raises:
            RuntimeError: _description_
        """
        self.world_size = tp * dp * pp * cp
        self.name_to_size = {
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "cp": cp,
        }
        order = "tp-pp-cp-dp"
        self.order = order
        order = order.lower()

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:
                order = order + "-" + name
        self.order = "-".join([token for token in order.split("-")])
        self.ordered_size = []

        for token in order.split("-"):
            self.ordered_size.append(self.name_to_size[token])

    def _get_mask(self, order: str, token: str):
        ordered_token = order.split("-")
        token = token.split("-")
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_replica_groups(self, token):
        """Get all replica groups by input token.

        Arguments:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.
        """
        parallel_size = self.ordered_size
        order = self.order
        mask = self._get_mask(order, token)
        ranks = generate_masked_orthogonal_rank_groups(
            self.world_size, parallel_size, mask
        )
        return ranks
    
    def get_rank_replica_group(self, token, rank):
        """Get the replica group that the given rank belongs to

        Args:
            token (_type_): _description_
            rank (_type_): _description_
        """
        all_groups = self.get_replica_groups(token)
        rank_group = None
        for ranks in all_groups:
            if rank in ranks:
                rank_group = ranks
        assert rank_group is not None, f"Rank {rank} does not belong to any of the {token} groups."
        return rank_group