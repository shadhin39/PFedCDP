import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_loaders import load_cbis_ddsm_dataset
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CBISDDSMTrainer:
    def __init__(self, image_size=(224, 224), batch_size=32, epochs=50):
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.models = {}
        self.histories = {}
        
    def load_dataset(self):
        """
        Load CBIS-DDSM dataset
        """
        print("Loading CBIS-DDSM dataset...")
        x_train, y_train, x_test, y_test = load_cbis_ddsm_dataset()
        
        self.data = {
            'x_train': x_train, 'y_train': y_train,
            'x_test': x_test, 'y_test': y_test,
            'num_classes': 2  # Benign, Malignant
        }
        
        print(f"CBIS-DDSM Dataset: Train {x_train.shape}, Test {x_test.shape}")
        print(f"Classes: Benign (0), Malignant (1)")
        
    def create_cnn_model(self, input_shape, num_classes, model_name="basic"):
        """
        Create CNN models with different architectures for CBIS-DDSM
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
                layers.Dense(num_classes, activation='sigmoid')  # Binary classification
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
                layers.Dense(num_classes, activation='sigmoid')  # Binary classification
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
                layers.Dense(num_classes, activation='sigmoid')  # Binary classification
            ])
            
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Compile model for binary classification
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        return model
    
    def train_model(self, model, model_name, validation_split=0.2):
        """
        Train a model with CBIS-DDSM data
        """
        print(f"\nTraining {model_name} model on CBIS-DDSM dataset...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                f'cbis_ddsm_{model_name}_best_model.h5',
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
        Evaluate model performance on CBIS-DDSM test set
        """
        print(f"\nEvaluating {model_name} model on CBIS-DDSM test set...")
        
        # Test evaluation
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
            self.data['x_test'], self.data['y_test'], verbose=0
        )
        
        # Predictions
        predictions = model.predict(self.data['x_test'])
        y_pred = (predictions > 0.5).astype(int).flatten()
            
        # Classification report
        print(f"\n{model_name} Results on CBIS-DDSM:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.data['y_test'], y_pred, target_names=['Benign', 'Malignant']))
        
        return {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_loss': test_loss,
            'predictions': y_pred,
            'true_labels': self.data['y_test']
        }
    
    def plot_training_history(self, history, model_name):
        """
        Plot training history for CBIS-DDSM model
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'CBIS-DDSM {model_name} Training History', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'cbis_ddsm_{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, results, model_name):
        """
        Plot confusion matrix for CBIS-DDSM results
        """
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title(f'CBIS-DDSM {model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'cbis_ddsm_{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_training(self):
        """
        Main function to run centralized training on CBIS-DDSM dataset
        """
        print("Starting Centralized Training on CBIS-DDSM Dataset")
        print("=" * 60)
        
        # Load dataset
        self.load_dataset()
        
        # Training configurations
        model_types = ['basic', 'advanced', 'lightweight']
        
        # Train all models
        all_results = {}
        
        for model_type in model_types:
            print(f"\n{'='*50}")
            print(f"Training: CBIS-DDSM {model_type.upper()} Model")
            print(f"{'='*50}")
            
            # Create and compile model
            input_shape = self.data['x_train'].shape[1:]
            num_classes = self.data['num_classes']
            
            model = self.create_cnn_model(input_shape, num_classes, model_type)
            model = self.compile_model(model)
            
            print(f"Model architecture for CBIS-DDSM {model_type}:")
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
            model.save(f"cbis_ddsm_{model_type}_final_model.h5")
            print(f"Model saved as cbis_ddsm_{model_type}_final_model.h5")
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def generate_comparison_report(self, all_results):
        """
        Generate a comparison report of all CBIS-DDSM models
        """
        print("\n" + "="*60)
        print("CBIS-DDSM CENTRALIZED TRAINING COMPARISON REPORT")
        print("="*60)
        
        comparison_data = []
        for model_type, result in all_results.items():
            comparison_data.append({
                'Model': f'CBIS-DDSM {model_type.upper()}',
                'Test Accuracy': f"{result['results']['test_accuracy']:.4f}",
                'Test Precision': f"{result['results']['test_precision']:.4f}",
                'Test Recall': f"{result['results']['test_recall']:.4f}",
                'Test Loss': f"{result['results']['test_loss']:.4f}"
            })
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)
        print("\nModel Performance Comparison:")
        print(df.to_string(index=False))
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'cbis_ddsm_training_results_{timestamp}.csv', index=False)
        print(f"\nResults saved to cbis_ddsm_training_results_{timestamp}.csv")

# Main execution
if __name__ == "__main__":
    # Create trainer instance
    trainer = CBISDDSMTrainer(
        image_size=(224, 224),
        batch_size=16,  # Adjust based on GPU memory
        epochs=30
    )
    
    # Run centralized training
    results = trainer.run_training()
    
    print("\nCBIS-DDSM centralized training completed successfully!")
    print("Check the generated plots and saved models for detailed results.")