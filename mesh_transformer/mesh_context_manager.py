import jax
import jax.numpy as jnp
import numpy as np
import os
from einops import rearrange, repeat
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P


class MeshContextManager:
    def __init__(self, dp=1, mp=1, core=1):
        device_count = jax.device_count()
        if device_count == 1:
            # Single TPU core case
            self.mesh = Mesh(mesh_utils.create_device_mesh((1,)), axis_names=('single_core',))
        else:
            self.mesh = Mesh(mesh_utils.create_device_mesh((dp, mp, core)), axis_names=('dp', 'mp', 'core'))

    def get_mesh(self):
        return self.mesh

    def __enter__(self):
        return self.mesh.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.mesh.__exit__(exc_type, exc_val, exc_tb)

