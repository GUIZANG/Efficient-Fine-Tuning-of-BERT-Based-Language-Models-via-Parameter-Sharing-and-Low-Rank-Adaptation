import torch
print("CUDA:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

import evaluate
print(evaluate.__version__)

