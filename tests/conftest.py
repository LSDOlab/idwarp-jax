import os

# Set these before importing JAX; tests need no accelerator and exercise both dtypes.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
