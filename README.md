# Assignment
## This module provides a custom implementation of the core functionality of einops.rearrange using NumPy, with support for reshaping, transposing, axis splitting and merging, repeating, and ellipsis parsing.


---

### Design Decisions

1. Pattern Parsing:
We split the pattern using -> into input and output tokens. Tokens within parentheses indicate merged or split axes, and ellipsis (...) is expanded into uniquely named placeholder axes (e.g., b0, b1, …).

2. Axis Mapping:
The input shape is matched against flattened axis names to create a map from axis name to dimension size. This allows for reshaping and inference of dimensions during rearrangement.

3. Reshape and Transpose Logic:
The function first reshapes the tensor based on input pattern, then transposes the axes to match the output pattern, and finally reshapes again to achieve merged/split outputs.

4. Error Handling:
The implementation provides clear and helpful error messages for:

Invalid pattern syntax

Mismatched number of tensor dimensions and pattern tokens

Missing axis lengths for custom axis names

Duplicate or invalid axes in patterns

5. Efficiency:
The implementation minimizes intermediate reshaping and transposition operations, ensuring low overhead.

Approach

Step 1: Parse the pattern string into input and output tokens.

Step 2: Expand ellipsis (...) in both input and output.

Step 3: Flatten composite axes (e.g., (h w) → ['h', 'w']) for both input and output.

Step 4: Match the input pattern to the actual shape and create a shape map.

Step 5: Apply reshape and transpose based on the mapping from input to output.

Step 6: Final reshaping to match requested output axes
How to Run the Module

Installation Requirements

No external dependencies are needed beyond NumPy:

pip install numpy

Usage Example

import numpy as np
from your_module import rearrange

# Example 1: Transpose
x = np.random.rand(3, 4)
out = rearrange(x, 'h w -> w h')

# Example 2: Merge axes
x = np.random.rand(3, 4, 5)
out = rearrange(x, 'a b c -> (a b) c')

# Example 3: Split axis
x = np.random.rand(12, 10)
out = rearrange(x, '(h w) c -> h w c', h=3)

# Example 4: Repeat axis
x = np.random.rand(3, 1, 5)
out = rearrange(x, 'a 1 c -> a b c', b=4)

# Example 5: Ellipsis handling
x = np.random.rand(2, 3, 4, 5)
out = rearrange(x, '... h w -> ... (h w)')

