import tensorflow as tf

class KnowledgeDistillationLoss:
    """Knowledge Distillation Loss Function for client-side training"""
    
    def __init__(self, temperature=3.0, lambda_balance=0.5):
        self.temperature = temperature
        self.lambda_balance = lambda_balance
        
    def compute_distillation_loss(self, student_logits, teacher_logits, true_labels):
        """Compute combined distillation and CE loss"""
        # Convert true labels to one-hot if needed
        if len(true_labels.shape) == 1:
            true_labels_onehot = tf.one_hot(true_labels, depth=10)
        else:
            true_labels_onehot = true_labels
            
        # Softmax with temperature for soft targets
        soft_teacher = tf.nn.softmax(teacher_logits / self.temperature)
        soft_student = tf.nn.softmax(student_logits / self.temperature)
        
        # KL divergence loss (teacher || student)
        kd_loss = tf.keras.losses.KLDivergence()(soft_teacher, soft_student)
        
        # Standard cross-entropy loss
        ce_loss = tf.keras.losses.categorical_crossentropy(
            true_labels_onehot, tf.nn.softmax(student_logits)
        )
        ce_loss = tf.reduce_mean(ce_loss)
        
        # Combined loss with temperature scaling
        total_loss = ((1 - self.lambda_balance) * ce_loss + 
                     self.lambda_balance * (self.temperature ** 2) * kd_loss)
        
        return total_loss, ce_loss, kd_loss