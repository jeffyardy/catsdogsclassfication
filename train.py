
import os, sys

# guard against a local mlflow.py or stale module causing circular import issues
if os.path.exists('mlflow.py'):
    raise RuntimeError("Local file named 'mlflow.py' found in working directory; rename or remove it to avoid import conflicts")
# clear any previously imported module
sys.modules.pop('mlflow', None)

import mlflow
# verify we have the full package (mlflow-skinny doesn't include mlflow.store)
if not hasattr(mlflow, 'store'):
    raise ImportError(
        "The installed 'mlflow' module is incomplete (mlflow-skinny or damaged).\n"
        "Install the full mlflow package: `pip install mlflow==2.15.1`"
    )

import mlflow.tensorflow
import tensorflow as tf
import numpy as np
from src.model import build_model

mlflow.set_experiment("cats_dogs")

# dummy random data placeholder (replace with real pipeline)
X = np.random.rand(20,224,224,3)
y = np.random.randint(0,2,20)

with mlflow.start_run():
    model = build_model()
    history = model.fit(X,y,epochs=1,validation_split=0.2)
    model.save("model.h5")
    mlflow.log_artifact("model.h5")
    mlflow.log_metric("final_acc", history.history["accuracy"][-1])
