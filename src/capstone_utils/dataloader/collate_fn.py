import torch


def minibatch_padding_collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """
    Collate function for the dataloader. This function pads the sequences in the batch to have the same length.
    :param batch: list of samples
    :return: padded batch
    """
    is_tuple = isinstance(batch[0], tuple)
    if is_tuple:
        # Get the maximum sequence length
        max_length = max([sample[1].shape[0] for sample in batch])
        max_length = (max_length // 4) * 4

        # Pad the sequences

        padded_batch = torch.zeros((len(batch), max_length), dtype=torch.long)
        for i, (_, sample) in enumerate(batch):
            padded_batch[i, : sample.shape[0]] = sample[:max_length]

        return torch.stack([sample[0] for sample in batch]), padded_batch

    else:
        # Get the maximum sequence length
        max_length = max([sample.shape[0] for sample in batch])
        max_length = (max_length // 4) * 4

        # Pad the sequences
        padded_batch = torch.zeros((len(batch), max_length, batch[0].shape[1]))
        for i, sample in enumerate(batch):
            padded_batch[i, : sample.shape[0]] = sample[:max_length]

        return padded_batch
