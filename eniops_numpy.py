import numpy as np
import re
from typing import List, Dict, Tuple

def parse_pattern(pattern: str) -> Tuple[List[str], List[str]]:
    """Parses the einops-style pattern and returns input/output axis info."""
    pattern = pattern.replace("...", "batch")  
    parts = pattern.split("->")
    if len(parts) != 2:
        raise ValueError("Pattern must have exactly one '->' separator.")
    input_pattern, output_pattern = map(str.strip, parts)
    return re.findall(r'\w+|\(\w+(?: \w+)*\)', input_pattern), re.findall(r'\w+|\(\w+(?: \w+)*\)', output_pattern)

def get_shape_from_pattern(tensor_shape: Tuple[int, ...], input_axes: List[str], axes_lengths: Dict[str, int]) -> Dict[str, int]:
    """Returns a mapping of axis names to their corresponding lengths."""
    shape_dict = {}
    dim_idx = 0
    for axis in input_axes:
        if '(' in axis:  # Merged axes
            merged_axes = re.findall(r'\w+', axis)
            merged_size = np.prod([axes_lengths.get(ax, tensor_shape[dim_idx + i]) for i, ax in enumerate(merged_axes)])
            shape_dict[axis] = int(merged_size)
            dim_idx += len(merged_axes)
        else:
            if dim_idx >= len(tensor_shape):
                raise IndexError("Too many dimensions specified in pattern.")
            shape_dict[axis] = tensor_shape[dim_idx]
            dim_idx += 1
    return shape_dict

def expand_pattern(output_axes: List[str], shape_dict: Dict[str, int], axes_lengths: Dict[str, int]) -> List[int]:
    """Expands the output pattern by replacing dimensions with actual sizes."""
    expanded_shape = []
    for axis in output_axes:
        if '(' in axis:  # Merging axes
            merged_axes = re.findall(r'\w+', axis)
            merged_size = np.prod([axes_lengths.get(ax, shape_dict.get(ax, 1)) for ax in merged_axes])
            expanded_shape.append(int(merged_size))
        elif axis in shape_dict:
            expanded_shape.append(shape_dict[axis])
        elif axis in axes_lengths:  # Repeating axis
            expanded_shape.append(axes_lengths[axis])
        else:
            raise ValueError(f"Unknown axis '{axis}' in output pattern")
    return expanded_shape

def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    """Rearranges a NumPy array according to an einops-style pattern."""
    input_axes, output_axes = parse_pattern(pattern)
    shape_dict = get_shape_from_pattern(tensor.shape, input_axes, axes_lengths)
    expanded_shape = expand_pattern(output_axes, shape_dict, axes_lengths)

    # Reshape according to input pattern (handling merged axes)
    reshape_dims = []
    for ax in input_axes:
        if '(' in ax:  # Handle merged axes
            merged_axes = re.findall(r'\w+', ax)
            reshape_dims.append(np.prod([shape_dict[merged_ax] for merged_ax in merged_axes]))
        else:
            reshape_dims.append(shape_dict[ax])
    
    tensor = tensor.reshape(*reshape_dims)

    # Determine valid axis order for transposition
    axis_order = [input_axes.index(ax) for ax in output_axes if ax in input_axes]
    tensor = np.transpose(tensor, axes=axis_order)

    # Reshape to final output shape
    return tensor.reshape(*expanded_shape)

# Example tests
if __name__ == "__main__":
    x = np.random.rand(3, 4)
    print(rearrange(x, 'h w -> w h').shape)  # (4, 3)

    x = np.random.rand(12, 10)
    print(rearrange(x, '(h w) c -> h w c', h=3, w=4).shape)  # (3, 4, 10)

    x = np.random.rand(3, 4, 5)
    print(rearrange(x, 'a b c -> (a b) c').shape)  # (12, 5)

    x = np.random.rand(3, 1, 5)
    print(rearrange(x, 'a 1 c -> a b c', b=4))