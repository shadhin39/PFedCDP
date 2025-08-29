import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_loaders import load_mias_dataset
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MIASTrainer:
    def __init__(self, image_size=(224, 224), batch_size=32, epochs=50):
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.models = {}
        self.histories = {}
        
    def load_dataset(self):
        """
        Load MIAS dataset
        """
        print("Loading MIAS dataset...")
        x_train, y_train, x_test, y_test = load_mias_dataset()
        
        self.data = {
            'x_train': x_train, 'y_train': y_train,
            'x_test': x_test, 'y_test': y_test,
            'num_classes': 3  # Normal, Benign, Malignant
        }
        
        print(f"MIAS Dataset: Train {x_train.shape}, Test {x_test.shape}")
        print(f"Classes: Normal (0), Benign (1), Malignant (2)")
        
    def create_cnn_model(self, input_shape, num_classes, model_name="basic"):
        """
        Create CNN models with different architectures for MIAS
        """
        if model_name == "basic":
            model = keras.Sequential([
                keras.Input(shape=input_shape),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation='softmax')  # Multi-class classification
            ])
            
        elif model_name == "advanced":
            model = keras.Sequential([
                keras.Input(shape=input_shape),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation='softmax')  # Multi-class classification
            ])
            
        elif model_name == "lightweight":
            model = keras.Sequential([
                keras.Input(shape=input_shape),
                layers.Conv2D(16, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation='softmax')  # Multi-class classification
            ])
            
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Compile model for multi-class classification
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_model(self, model, model_name, validation_split=0.2):
        """
        Train a model with MIAS data
        """
        print(f"\nTraining {model_name} model on MIAS dataset...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                f'mias_{model_name}_best_model.h5',
                monitor='val_accuracy', save_best_only=True
            )
        ]
        
        # Train the model
        history = model.fit(
            self.data['x_train'], self.data['y_train'],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, model_name):
        """
        Evaluate model performance on MIAS test set with comprehensive metrics
        """
        print(f"\nEvaluating {model_name} model on MIAS test set...")
        
        # Test evaluation
        test_loss, test_accuracy = model.evaluate(
            self.data['x_test'], self.data['y_test'], verbose=0
        )
        
        # Predictions
        predictions = model.predict(self.data['x_test'])
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate comprehensive metrics using sklearn
        sklearn_accuracy = accuracy_score(self.data['y_test'], y_pred)
        precision = precision_score(self.data['y_test'], y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.data['y_test'], y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.data['y_test'], y_pred, average='weighted', zero_division=0)
        
        # AUC Score for multi-class (using one-vs-rest)
        try:
            auc_score = roc_auc_score(self.data['y_test'], predictions, multi_class='ovr', average='weighted')
        except ValueError:
            auc_score = 0.0  # Handle case where classes are missing
        
        # Calculate specificity and sensitivity for multi-class
        cm = confusion_matrix(self.data['y_test'], y_pred)
        
        # For multi-class, calculate macro-averaged specificity and sensitivity
        specificities = []
        sensitivities = []
        
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp
            
            specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            specificities.append(specificity_i)
            sensitivities.append(sensitivity_i)
        
        avg_specificity = np.mean(specificities)
        avg_sensitivity = np.mean(sensitivities)
            
        # Classification report
        print(f"\n{model_name} Results on MIAS:")
        print(f"Test Accuracy: {test_accuracy:.4f} (TF) / {sklearn_accuracy:.4f} (sklearn)")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test AUC Score: {auc_score:.4f}")
        print(f"Test Specificity: {avg_specificity:.4f}")
        print(f"Test Sensitivity: {avg_sensitivity:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.data['y_test'], y_pred, target_names=['Normal', 'Benign', 'Malignant']))
        
        return {
            'test_accuracy': sklearn_accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1_score': f1,
            'test_auc_score': auc_score,
            'test_specificity': avg_specificity,
            'test_sensitivity': avg_sensitivity,
            'test_loss': test_loss,
            'predictions': y_pred,
            'true_labels': self.data['y_test']
        }
    
    def plot_training_history(self, history, model_name):
        """
        Plot training history for MIAS model
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'MIAS {model_name} Training History', fontsize=16)
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        # Loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'mias_{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, results, model_name):
        """
        Plot confusion matrix for MIAS results
        """
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Benign', 'Malignant'], 
                   yticklabels=['Normal', 'Benign', 'Malignant'])
        plt.title(f'MIAS {model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'mias_{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_training(self):
        """
        Main function to run centralized training on MIAS dataset
        """
        print("Starting Centralized Training on MIAS Dataset")
        print("=" * 50)
        
        # Load dataset
        self.load_dataset()
        
        # Training configurations
        model_types = ['basic', 'advanced', 'lightweight']
        
        # Train all models
        all_results = {}
        
        for model_type in model_types:
            print(f"\n{'='*50}")
            print(f"Training: MIAS {model_type.upper()} Model")
            print(f"{'='*50}")
            
            # Create and compile model
            input_shape = self.data['x_train'].shape[1:]
            num_classes = self.data['num_classes']
            
            model = self.create_cnn_model(input_shape, num_classes, model_type)
            model = self.compile_model(model)
            
            print(f"Model architecture for MIAS {model_type}:")
            model.summary()
            
            # Train model
            history = self.train_model(model, model_type)
            
            # Evaluate model
            results = self.evaluate_model(model, model_type)
            
            # Store results
            all_results[model_type] = {
                'model': model,
                'history': history,
                'results': results
            }
            
            # Plot results
            self.plot_training_history(history, model_type)
            self.plot_confusion_matrix(results, model_type)
            
            # Save model
            model.save(f"mias_{model_type}_final_model.h5")
            print(f"Model saved as mias_{model_type}_final_model.h5")
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def generate_comparison_report(self, all_results):
        """
        Generate a comparison report of all MIAS models
        """
        print("\n" + "="*50)
        print("MIAS CENTRALIZED TRAINING COMPARISON REPORT")
        print("="*50)
        
        comparison_data = []
        for model_type, result in all_results.items():
            comparison_data.append({
                'Model': f'MIAS {model_type.upper()}',
                'Test Accuracy': f"{result['results']['test_accuracy']:.4f}",
                'Test Precision': f"{result['results']['test_precision']:.4f}",
                'Test Recall': f"{result['results']['test_recall']:.4f}",
                'Test F1 Score': f"{result['results']['test_f1_score']:.4f}",
                'Test AUC Score': f"{result['results']['test_auc_score']:.4f}",
                'Test Specificity': f"{result['results']['test_specificity']:.4f}",
                'Test Sensitivity': f"{result['results']['test_sensitivity']:.4f}",
                'Test Loss': f"{result['results']['test_loss']:.4f}"
            })
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)
        print("\nModel Performance Comparison:")
        print(df.to_string(index=False))
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'mias_training_results_{timestamp}.csv', index=False)
        print(f"\nResults saved to mias_training_results_{timestamp}.csv")

# Main execution
if __name__ == "__main__":
    # Create trainer instance
    trainer = MIASTrainer(
        image_size=(224, 224),
        batch_size=16,  # Adjust based on GPU memory
        epochs=30
    )
    
    # Run centralized training
    results = trainer.run_training()
    
    print("\nMIAS centralized training completed successfully!")
    print("Check the generated plots and saved models for detailed results.")
