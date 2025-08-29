import tensorflow as tf
import numpy as np

class Quantization:
    """Quantization implementation for communication efficiency"""
    
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.max_val = 2**num_bits - 1
    
    def quantize_weights(self, weights):
        """Quantize weights to lower-bit representation"""
        print(f"[QUANTIZATION] Quantizing weights to {self.num_bits} bits")
        
        quantized_weights = []
        quantization_params = []
        
        for w in weights:
            # Find min and max values
            w_min = tf.reduce_min(w)
            w_max = tf.reduce_max(w)
            
            # Quantize: scale to [0, 2^b - 1] and round
            w_scaled = (w - w_min) / (w_max - w_min + 1e-8)  # Scale to [0, 1]
            w_quantized = tf.round(w_scaled * self.max_val)  # Scale to [0, 2^b-1] and round
            
            # Convert to integers
            w_quantized = tf.cast(w_quantized, tf.int32)
            
            quantized_weights.append(w_quantized.numpy())
            quantization_params.append({
                'min': float(w_min.numpy()),
                'max': float(w_max.numpy())
            })
        
        print(f"[QUANTIZATION] Quantization completed")
        return quantized_weights, quantization_params
    
    def dequantize_weights(self, quantized_weights, quantization_params):
        """Dequantize weights back to float representation"""
        print(f"[QUANTIZATION] Dequantizing weights from {self.num_bits} bits")
        
        dequantized_weights = []
        
        for q_w, params in zip(quantized_weights, quantization_params):
            # Convert back to float and scale
            w_scaled = tf.cast(q_w, tf.float32) / self.max_val  # Scale to [0, 1]
            w_dequantized = w_scaled * (params['max'] - params['min']) + params['min']
            dequantized_weights.append(w_dequantized)
        
        print(f"[QUANTIZATION] Dequantization completed")
        return dequantized_weights