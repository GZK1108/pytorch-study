import torch.nn as nn
import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)

class trans(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, src):
        src1 = self.fc(src)
        src2 = nn.Linear(10, 2)(src)
        three = nn.Linear(10, 2)
        src3 = three(src)
        return src1, src2, src3


src = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) # (batch_size, seq_len)
model = trans()
out1,out2,out3 = model(src)
print(out1,out2,out3)