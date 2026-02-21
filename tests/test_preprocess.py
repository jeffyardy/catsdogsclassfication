
from src.preprocess import batch_stack
import numpy as np

def test_batch_stack():
    a=np.zeros((224,224,3))
    b=np.zeros((224,224,3))
    out=batch_stack([a,b])
    assert out.shape==(2,224,224,3)
