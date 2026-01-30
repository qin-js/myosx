import sys
import numpy as np
import os

# 1. 查看Python解释器和路径
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# 2. 查看实际加载的NumPy路径
print(f"NumPy version: {np.__version__}")
print(f"NumPy location: {np.__file__}")

# 3. 检查sys.path中的所有路径
print("Python search paths:")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")
    # 检查每个路径中是否有numpy
    if os.path.exists(os.path.join(path, 'numpy')):
        print(f"  *** NumPy found in this path ***")