import tensorflow as tf
import numpy as np

class FisherInformationCalculator:
    """Calculate Fisher Information Matrix for feature extractor personalization"""
    
    def __init__(self, threshold=0.01):
        self.threshold = threshold
    
    def calculate_fisher_information(self, model, x_data, y_data, batch_size=32):
        """Calculate Fisher Information Matrix for model parameters"""
        print(f"[FISHER] Calculating Fisher Information Matrix...")
        
        # Ensure proper data format
        if len(y_data.shape) == 1:
            y_data_onehot = tf.one_hot(y_data, depth=10)
        else:
            y_data_onehot = y_data
        
        # Create dataset with proper error handling
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data_onehot))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Initialize Fisher information storage
        fisher_info = []
        for layer in model.trainable_variables:
            fisher_info.append(tf.zeros_like(layer))
        
        num_samples = 0
        
        # Calculate Fisher information over batches with proper error handling
        try:
            for batch_x, batch_y in dataset:
                try:
                    with tf.GradientTape() as tape:
                        predictions = model(batch_x, training=False)
                        loss = tf.keras.losses.categorical_crossentropy(batch_y, predictions)
                        loss = tf.reduce_mean(loss)
                    
                    # Calculate gradients
                    gradients = tape.gradient(loss, model.trainable_variables)
                    
                    # Accumulate squared gradients (Fisher Information approximation)
                    for i, grad in enumerate(gradients):
                        if grad is not None:
                            fisher_info[i] += tf.square(grad) * batch_x.shape[0]
                    
                    num_samples += batch_x.shape[0]
                    
                except Exception as batch_error:
                    print(f"[FISHER] Batch processing error: {batch_error}")
                    continue
                    
        except Exception as dataset_error:
            print(f"[FISHER] Dataset processing error: {dataset_error}")
            return fisher_info
        
        # Normalize by number of samples
        if num_samples > 0:
            for i in range(len(fisher_info)):
                fisher_info[i] = fisher_info[i] / num_samples
        
        print(f"[FISHER] Fisher Information calculated for {num_samples} samples")
        return fisher_info
    
    def create_binary_masks(self, fisher_info):
        """Create binary masks based on Fisher Information threshold"""
        print(f"[FISHER] Creating binary masks with threshold {self.threshold}")
        
        personal_masks = []
        global_masks = []
        
        total_params = 0
        high_fisher_params = 0
        
        for fisher_layer in fisher_info:
            # Create binary mask: 1 for high Fisher info (keep local), 0 for low Fisher info (use global)
            personal_mask = tf.cast(fisher_layer > self.threshold, tf.float32)
            global_mask = 1.0 - personal_mask
            
            personal_masks.append(personal_mask)
            global_masks.append(global_mask)
            
            # Statistics
            layer_total = tf.size(fisher_layer).numpy()
            layer_high_fisher = tf.reduce_sum(personal_mask).numpy()
            
            total_params += layer_total
            high_fisher_params += layer_high_fisher
            
            print(f"[FISHER] Layer shape {fisher_layer.shape}: {layer_high_fisher}/{layer_total} ({100*layer_high_fisher/layer_total:.1f}%) high Fisher params")
        
        print(f"[FISHER] Total: {high_fisher_params}/{total_params} ({100*high_fisher_params/total_params:.1f}%) parameters kept locally")
        
        return personal_masks, global_masks
    
    def apply_fisher_personalization(self, local_weights, global_weights, personal_masks, global_masks):
        """Apply Fisher-based personalization to combine local and global weights"""
        personalized_weights = []
        
        for i, (local_w, global_w, personal_mask, global_mask) in enumerate(
            zip(local_weights, global_weights, personal_masks, global_masks)):
            
            # Combine weights using masks: W_L,k^t = M_personal ⊙ W_L,k^{t-1} + M_global ⊙ W_L,global^{t-1}
            personalized_w = personal_mask * local_w + global_mask * global_w
            personalized_weights.append(personalized_w)
        
        return personalized_weights