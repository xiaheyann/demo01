import time
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import torch_npu
    torch.backends.mha.set_fastpath_enabled(False)
    DEVICE = torch.device("npu")
except:
    DEVICE = torch.device("cpu")

DEVICE = torch.device("cpu")

class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        vocab_size=10000,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, input_ids, padding_mask):

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        pos_ids = (
            torch.arange(seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)

        out = self.encoder(x, src_key_padding_mask=padding_mask)

        return out


def generate_variable_batch(batch_size, max_seq_len, vocab_size, device):

    lengths = torch.randint(
        low=max_seq_len // 4,
        high=max_seq_len,
        size=(batch_size,),
    )

    max_len = lengths.max().item()

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, l in enumerate(lengths):
        input_ids[i, :l] = torch.randint(0, vocab_size, (l,))
        padding_mask[i, :l] = False

    return input_ids.to(device), padding_mask.to(device)


def main():

    vocab_size = 10000
    batch_size = 32
    max_seq_len = 256
    steps = 100

    print("device:", DEVICE)

    model = TransformerEncoderModel(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        max_len=max_seq_len,
    ).to(DEVICE)

    model.eval()

    # warmup
    with torch.inference_mode():
        for _ in range(10):
            input_ids, padding_mask = generate_variable_batch(
                batch_size,
                max_seq_len,
                vocab_size,
                DEVICE,
            )
            _ = model(input_ids, padding_mask)

    if DEVICE.type == "npu":
        torch.npu.synchronize()

    start = time.time()

    with torch.inference_mode():

        for _ in tqdm(range(steps), desc="Inference"):

            input_ids, padding_mask = generate_variable_batch(
                batch_size,
                max_seq_len,
                vocab_size,
                DEVICE,
            )

            output = model(input_ids, padding_mask)

    if DEVICE.type == "npu":
        torch.npu.synchronize()

    end = time.time()

    print("output shape:", output.shape)
    print("total time:", end - start)
    print("avg latency:", (end - start) / steps)


if __name__ == "__main__":
    main()
