import numpy as np
import logging
import matplotlib.pyplot as plt
import torch


logging.basicConfig(level=logging.INFO)
array_test = np.arange(12)
array_reshape = np.random.randint(5, 30, 12).reshape(3, 4)

reshape_test = array_test.reshape(3, 4)
logging.info(reshape_test)

result = np.stack(arrays=[array_reshape, reshape_test], axis=0)
con = np.concatenate([array_reshape, reshape_test], axis=-1 )
logging.info(result)
logging.info(con)

scores = np.random.randint(0, 101, 50)
plt.figure()
plt.plot(scores - scores.mean())
plt.show()
# normalization
logging.info((scores - scores.min())/(scores.max() - scores.min()))

# standardize
plt.plot((scores - scores.mean()) / scores.std())

# pytorch: used as compute structure
# tensor: array
tensor = torch.tensor(data=[1, 2, 3])
torch.ones(6)
torch.rand(2, 3)
torch.randn(2, 3)
torch.randint(low=0, high=101, size=(2, 3))
torch.linspace(start=-5, end=50, steps=50)
t = torch.randn(3, 5)
logging.info(torch.cuda.is_available())

# move to gpu
device = ''
t.to(device)

