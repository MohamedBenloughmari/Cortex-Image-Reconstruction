import torch
import torch.nn as nn
from ConvLSTM.convlstm import ConvLSTM
# Create the ConvLSTM model
model = ConvLSTM(
    img_size=(20, 20),      # Height and width of input image
    input_dim=1,             # Number of input channels (since your tensor has 1 channel)
    hidden_dim=16,           # Number of hidden channels (you can adjust this)
    kernel_size=(3, 3),      # Kernel size for convolutions
    cnn_dropout=0.5,         # Dropout rate for CNN
    rnn_dropout=0.5,         # Dropout rate for RNN
    batch_first=True,        # Your input has batch first (B, T, C, H, W)
    bias=True,
    peephole=False,
    layer_norm=False,
    return_sequence=True,    # Return full sequence output
    bidirectional=False      # Set to True if you want bidirectional
)

# Your input tensor
x = torch.ones(size=(1, 10, 1, 20, 20))

# Forward pass
output, last_state, _ = model(x)

# Print the shapes
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Last state (h) shape: {last_state[0].shape}")
print(f"Last state (c) shape: {last_state[1].shape}")

# If you want to test with bidirectional model
print("\n" + "="*50)
print("Testing bidirectional model:")
print("="*50)

bidirectional_model = ConvLSTM(
    img_size=(20, 20),
    input_dim=1,
    hidden_dim=16,
    kernel_size=(3, 3),
    cnn_dropout=0.5,
    rnn_dropout=0.5,
    batch_first=True,
    bias=True,
    peephole=False,
    layer_norm=False,
    return_sequence=True,
    bidirectional=True       # Enable bidirectional
)

# Forward pass with bidirectional model
output_bi, last_state_fw, last_state_bw = bidirectional_model(x)

print(f"Input shape: {x.shape}")
print(f"Bidirectional output shape: {output_bi.shape}")  # Note: hidden_dim will be doubled (32)
print(f"Forward last state (h) shape: {last_state_fw[0].shape}")
print(f"Forward last state (c) shape: {last_state_fw[1].shape}")
print(f"Backward last state (h) shape: {last_state_bw[0].shape}")
print(f"Backward last state (c) shape: {last_state_bw[1].shape}")

# If you want to test without returning full sequence
print("\n" + "="*50)
print("Testing without returning sequence:")
print("="*50)

model_no_seq = ConvLSTM(
    img_size=(20, 20),
    input_dim=1,
    hidden_dim=16,
    kernel_size=(3, 3),
    cnn_dropout=0.5,
    rnn_dropout=0.5,
    batch_first=True,
    bias=True,
    peephole=False,
    layer_norm=False,
    return_sequence=False,   # Only return last output
    bidirectional=False
)

output_last, last_state, _ = model_no_seq(x)
print(f"Output shape (only last timestep): {output_last.shape}")  # Shape: (1, 1, 16, H, W)