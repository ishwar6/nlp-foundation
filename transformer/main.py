class Transformer(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.positional_encoding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.positional_encoding(x)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
