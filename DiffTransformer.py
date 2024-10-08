import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Applies normalization across the last dimension and scales the output.
    """
    def __init__(self, d, eps=1e-5):
        """
        Args:
            d (int): Dimension of the input features.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d).

        Returns:
            Tensor: Normalized and scaled tensor.
        """
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale

class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    Combines the Swish activation with Gated Linear Units.
    """
    def __init__(self, d_model):
        """
        Args:
            d_model (int): Dimension of the input features.
        """
        super().__init__()
        # Intermediate projection layers
        self.WG = nn.Linear(d_model, 8 * d_model // 3)
        self.W1 = nn.Linear(d_model, 8 * d_model // 3)
        self.W2 = nn.Linear(8 * d_model // 3, d_model)

    def forward(self, x):
        """
        Forward pass for SwiGLU.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying SwiGLU.
        """
        return self.W2(F.silu(self.WG(x)) * self.W1(x))

class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention Mechanism.
    Replaces the conventional softmax attention with a differential attention.
    """
    def __init__(self, d_model, num_heads, lambda_init):
        """
        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections for queries, keys, and values
        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)

        # Learnable parameters for lambda reparameterization
        self.lambda_q1 = nn.Parameter(torch.randn(self.d_head))
        self.lambda_k1 = nn.Parameter(torch.randn(self.d_head))
        self.lambda_q2 = nn.Parameter(torch.randn(self.d_head))
        self.lambda_k2 = nn.Parameter(torch.randn(self.d_head))

        self.lambda_init = lambda_init

        # Scale parameter for RMSNorm
        self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self.eps = 1e-5  # Epsilon for numerical stability

    def forward(self, X):
        """
        Forward pass for Multi-Head Differential Attention.

        Args:
            X (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying differential attention.
        """
        batch, N, d_model = X.shape

        # Project inputs to queries, keys, and values
        Q = self.W_q(X)  # Shape: (batch, N, 2h*d)
        K = self.W_k(X)  # Shape: (batch, N, 2h*d)
        V = self.W_v(X)  # Shape: (batch, N, 2h*d)

        # Reshape and permute for multi-head attention
        Q = Q.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)  # (batch, h, N, 2d)
        K = K.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)  # (batch, h, N, 2d)
        V = V.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)  # (batch, h, N, 2d)

        # Split Q and K into Q1, Q2 and K1, K2
        Q1, Q2 = Q.chunk(2, dim=-1)  # Each of shape: (batch, h, N, d)
        K1, K2 = K.chunk(2, dim=-1)  # Each of shape: (batch, h, N, d)

        # Compute lambda using reparameterization
        # lambda = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init
        lambda_q1_dot_k1 = torch.dot(self.lambda_q1, self.lambda_k1)
        lambda_q2_dot_k2 = torch.dot(self.lambda_q2, self.lambda_k2)
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init  # Scalar

        # Compute attention scores
        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling  # (batch, h, N, N)
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling  # (batch, h, N, N)

        # Apply softmax to get attention weights
        attention1 = F.softmax(A1, dim=-1)
        attention2 = F.softmax(A2, dim=-1)
        attention = attention1 - lambda_val * attention2  # (batch, h, N, N)

        # Apply attention weights to values
        O = torch.matmul(attention, V)  # (batch, h, N, 2d)

        # Normalize each head independently using RMSNorm
        O_reshaped = O.reshape(batch * self.num_heads, N, 2 * self.d_head)  # (batch*h, N, 2d)
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (batch*h, N, 1)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale  # (batch*h, N, 2d)

        # Reshape back to (batch, h, N, 2d)
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head)

        # Scale the normalized output
        O_normalized = O_normalized * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        O_concat = O_normalized.transpose(1, 2).reshape(batch, N, self.num_heads * 2 * self.d_head)  # (batch, N, 2d*h)

        # Final linear projection
        out = self.W_o(O_concat)  # (batch, N, d_model)

        return out

class DiffTransformerLayer(nn.Module):
    """
    Single Layer of the DiffTransformer Architecture.
    Consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.
    """
    def __init__(self, d_model, num_heads, lambda_init):
        """
        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda in Differential Attention.
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadDifferentialAttention(d_model, num_heads, lambda_init)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model)

    def forward(self, x):
        """
        Forward pass for a single transformer layer.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after processing through the layer.
        """
        # Apply Multi-Head Differential Attention with residual connection
        y = self.attn(self.norm1(x)) + x
        # Apply SwiGLU Feed-Forward Network with residual connection
        z = self.ff(self.norm2(y)) + y
        return z

class DiffTransformer(nn.Module):
    """
    The DiffTransformer Model incorporating multiple DiffTransformerLayers.
    Suitable for sequence modeling tasks such as language modeling.
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_length=512):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            max_seq_length (int): Maximum sequence length.
        """
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_length, d_model)
        self.layers = nn.ModuleList([
            DiffTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                lambda_init=0.8 - 0.6 * math.exp(-0.3 * (l - 1))
            )
            for l in range(1, num_layers + 1)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        """
        Forward pass for the DiffTransformer.

        Args:
            x (Tensor): Input tensor of token indices of shape (batch, sequence_length).

        Returns:
            Tensor: Logits for each token in the vocabulary of shape (batch, sequence_length, vocab_size).
        """
        batch, N = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(batch, N)  # (batch, N)
        X = self.token_emb(x) + self.pos_emb(positions)  # (batch, N, d_model)

        for layer in self.layers:
            X = layer(X)

        X = self.norm(X)  # (batch, N, d_model)
        logits = self.head(X)  # (batch, N, vocab_size)
        return logits

# Example usage:
if __name__ == "__main__":
    # Define model hyperparameters
    vocab_size = 30522  # Example vocabulary size (e.g., BERT)
    d_model = 768
    num_heads = 12
    num_layers = 12
    max_seq_length = 512

    # Instantiate the model
    model = DiffTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_length=max_seq_length
    )

    # Example input: batch of token indices
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))  # (batch, N)

    # Forward pass
    logits = model(input_ids)  # (batch, N, vocab_size)
    print(logits.shape)  # Should output: torch.Size([2, 128, 30522])
