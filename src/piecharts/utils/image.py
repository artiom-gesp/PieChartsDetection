import torch
import torch.nn.functional as F
from jaxtyping import Shaped


def pad_tensor_to_divisible_by_N(tensor: Shaped[torch.Tensor, "... X Y"], N: int = 16, mode: str = "reflect", value: float = 0):
    # Get current height and width
    height, width = tensor.shape[-2:]

    # Calculate the padding needed to make the width and height divisible by N
    pad_height = (N - height % N) % N
    pad_width = (N - width % N) % N

    # Define padding: pad the last dimension by (left, right) and second-last dimension by (top, bottom)
    padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)

    # Pad the tensor and return
    padded_tensor = F.pad(tensor, padding, mode=mode, value=value)  # You can change 'constant' and `value` as needed

    return padded_tensor
