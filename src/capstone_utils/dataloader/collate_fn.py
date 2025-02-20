import torch


def minibatch_padding_collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """
    Collate function for the dataloader. This function pads the sequences in the batch to have the same length.
    :param batch: list of samples
    :return: padded batch
    """
    # Get the maximum sequence length
    max_length = max([sample.shape[0] for sample in batch])
    max_length = (max_length // 4) * 4

    # Pad the sequences
    padded_batch = torch.zeros((len(batch), max_length, batch[0].shape[1]))
    for i, sample in enumerate(batch):
        padded_batch[i, : sample.shape[0]] = sample[:max_length]

    return padded_batch
