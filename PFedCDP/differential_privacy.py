import tensorflow as tf
import numpy as np

class DifferentialPrivacy:
    """Differential Privacy implementation for federated learning"""
    
    def __init__(self, epsilon=8.0, delta=1e-5, clipping_threshold=2.5, noise_multiplier=None):
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_threshold = clipping_threshold
        self.noise_multiplier = noise_multiplier or self._compute_noise_multiplier()
    
    def _compute_noise_multiplier(self):
        """Compute noise multiplier based on privacy parameters"""
        # Simplified noise multiplier calculation
        # In practice, use privacy accountant for precise calculation
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def clip_weights(self, weights):
        """Clip weights using L2 norm clipping"""
        clipped_weights = []
        for w in weights:
            # Calculate L2 norm
            l2_norm = tf.norm(w, ord=2)
            # Apply clipping
            clipping_factor = tf.minimum(1.0, self.clipping_threshold / l2_norm)
            clipped_w = w * clipping_factor
            clipped_weights.append(clipped_w)
        return clipped_weights
    
    def add_noise(self, weights):
        """Add calibrated Gaussian noise for differential privacy"""
        noisy_weights = []
        for w in weights:
            # Generate Gaussian noise
            noise_stddev = self.clipping_threshold * self.noise_multiplier
            noise = tf.random.normal(shape=w.shape, mean=0.0, stddev=noise_stddev)
            noisy_w = w + noise
            noisy_weights.append(noisy_w)
        return noisy_weights
    
    def apply_dp(self, weights):
        """Apply differential privacy: clipping + noise addition"""
        print(f"[DP] Applying differential privacy with ε={self.epsilon}, δ={self.delta}")
        
        # Step 1: Clip weights
        clipped_weights = self.clip_weights(weights)
        
        # Step 2: Add noise
        dp_weights = self.add_noise(clipped_weights)
        
        print(f"[DP] Differential privacy applied successfully")
        return dp_weights