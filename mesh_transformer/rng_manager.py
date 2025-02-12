import jax
import jax.numpy as jnp

class RNGManager:
    def __init__(self, seed=42):
        """
        Initialize the RNGManager with a base seed.
        """
        self.base_rng = jax.random.PRNGKey(seed)
        self.current_rng = self.base_rng

    def split_keys(self, num_splits):
        if jax.device_count() == 1:
            return self.base_rng  # Use single key when there's only one device
        else:
            split_keys = jax.random.split(self.base_rng, num_splits)
            self.base_rng = split_keys[0]  # Update base_rng to the next key in sequence
            return split_keys


    def get_current_key(self):
        """
        Get the current RNG key.
        """
        return self.base_rng

    def reset(self, seed=None):
        """
        Reset the RNG to the initial seed or a new one.
        """
        if seed is not None:
            self.base_rng = jax.random.PRNGKey(seed)
        self.current_rng = self.base_rng
