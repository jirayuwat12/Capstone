import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 32,
        codebook_dim: int = 256,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        reset_threshold: int = 1,
    ) -> None:
        super(Quantizer, self).__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.epsilon = epsilon
        self.reset_threshold = reset_threshold  # Minimum usage threshold before resetting a codebook vector

        # Initialize embedding vectors
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # Commitment cost hyperparameter
        self.commitment_cost = 0.25

        # EMA updates for embeddings
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embedding_avg", self.embedding.weight.clone())

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize the input tensor x

        Args:
            x (torch.Tensor): Input tensor of shape (B, embed_dim, T/l)

        Returns:
            quantized (torch.Tensor): Quantized tensor
            loss (torch.Tensor): Total quantization loss (commitment + embedding)
            encoding_indices (torch.Tensor): Codebook indices
        """
        # Reshape input tensor to (B*T/l, embed_dim)
        x = x.permute(0, 2, 1)  # (B, T/l, embed_dim)
        x_flatten = x.view(-1, self.codebook_dim)  # Flattened to (B*T/l, embed_dim)

        # Calculate distances between x and embeddings
        embedding = self.embedding.weight  # (codebook_size, embed_dim)
        distances = (
            torch.sum(x_flatten**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.matmul(x_flatten, embedding.t())
        )

        # Find the nearest embedding index for each input
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Create one-hot encoding for embeddings
        encodings = torch.zeros(encoding_indices.size(0), self.codebook_size, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the input
        quantized = torch.matmul(encodings, embedding).view(x.shape)

        # Compute losses
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), x)
        embedding_loss = F.mse_loss(quantized, x.detach())
        loss = commitment_loss + embedding_loss

        # Straight-through estimator for quantization
        quantized = x + (quantized - x).detach()

        # Exponential Moving Average (EMA) updates for embeddings
        if self.training:
            encodings_sum = encodings.sum(0)
            ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * encodings_sum

            # Laplace smoothing to avoid empty clusters
            n = torch.sum(ema_cluster_size) + self.epsilon
            self.ema_cluster_size = (ema_cluster_size + self.epsilon) / n * n

            # Update EMA embeddings
            embedding_sum = torch.matmul(encodings.t(), x_flatten)
            self.ema_embedding_avg = self.ema_embedding_avg * self.decay + (1 - self.decay) * embedding_sum

            # Normalize the embedding weights
            self.embedding.weight.data.copy_(self.ema_embedding_avg / self.ema_cluster_size.unsqueeze(1))

            # Reset unused codebook vectors
            self._reset_codebook_vectors()

        # Reshape encoding indices to match input
        encoding_indices = encoding_indices.view(x.size(0), x.size(1), 1)

        return quantized, loss, encoding_indices

    def _reset_codebook_vectors(self):
        """
        Reset the codebook vectors that are used less than the threshold.
        This avoids codebook collapse by reinitializing rarely used vectors.
        """
        unused_codebooks = self.ema_cluster_size < self.reset_threshold
        num_resets = unused_codebooks.sum().item()

        if num_resets > 0:
            # Reset the embeddings for these unused codebooks
            with torch.no_grad():
                reset_indices = torch.nonzero(unused_codebooks).squeeze(1)
                self.embedding.weight.data[reset_indices] = (
                    torch.randn_like(self.embedding.weight.data[reset_indices]) * 0.01
                )

    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dequantize the input tensor x

        Args:
            x (torch.Tensor): Input tensor of shape (B, T/l, 1)

        Returns:
            x_dequantized (torch.Tensor): Dequantized tensor of shape (B, T/l, embed_dim)
        """
        return self.embedding(x.squeeze(2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.quantize(x)


if __name__ == "__main__":
    import torchsummary

    model = Quantizer()
    torchsummary.summary(model, (512, 16))