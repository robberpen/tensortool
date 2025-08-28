#!/usr/bin/env python
import numpy as np
import sys
# Load the .npy file
data = np.load(sys.argv[1])

# Print the shape and dtype
print(data.shape)
print(data.dtype)

# Access the content
print(data)

