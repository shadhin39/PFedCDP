import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_loaders import load_cbis_ddsm_dataset, load_mias_dataset
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CentralizedTrainer:
    def __init__(self, image_size=(224, 224), batch_size=32, epochs=50):
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.models = {}
        self.histories = {}
        
    def load_and_combine_datasets(self):
        """
        Load both CBIS-DDSM and MIAS datasets and combine them appropriately
        """
        print("Loading CBIS-DDSM dataset...")
        cbis_x_train, cbis_y_train, cbis_x_test, cbis_y_test = load_cbis_ddsm_dataset()
        
        print("Loading MIAS dataset...")
        mias_x_train, mias_y_train, mias_x_test, mias_y_test = load_mias_dataset()
        
        # Strategy 1: Train separate models for each dataset
        self.cbis_data = {
            'x_train': cbis_x_train, 'y_train': cbis_y_train,
            'x_test': cbis_x_test, 'y_test': cbis_y_test,
            'num_classes': 2  # Benign, Malignant
        }
        
        self.mias_data = {
            'x_train': mias_x_train, 'y_train': mias_y_train,
            'x_test': mias_x_test, 'y_test': mias_y_test,
            'num_classes': 3  # Normal, Benign, Malignant
        }
        
        # Strategy 2: Create combined dataset with unified labels
        # Map CBIS-DDSM labels to MIAS format (add Normal class)
        # For demonstration, we'll create a combined binary classification
        combined_x_train = np.concatenate([cbis_x_train, mias_x_train], axis=0)
        combined_x_test = np.concatenate([cbis_x_test, mias_x_test], axis=0)
        
        # Convert MIAS 3-class to binary (Normal+Benign=0, Malignant=1)
        mias_y_train_binary = np.where(mias_y_train == 2, 1, 0)
        mias_y_test_binary = np.where(mias_y_test == 2, 1, 0)
        
        combined_y_train = np.concatenate([cbis_y_train, mias_y_train_binary], axis=0)
        combined_y_test = np.concatenate([cbis_y_test, mias_y_test_binary], axis=0)
        
        self.combined_data = {
            'x_train': combined_x_train, 'y_train': combined_y_train,
            'x_test': combined_x_test, 'y_test': combined_y_test,
            'num_classes': 2  # Binary classification
        }
        
        print(f"CBIS-DDSM: Train {cbis_x_train.shape}, Test {cbis_x_test.shape}")
        print(f"MIAS: Train {mias_x_train.shape}, Test {mias_x_test.shape}")
        print(f"Combined: Train {combined_x_train.shape}, Test {combined_x_test.shape}")
        
    def create_cnn_model(self, input_shape, num_classes, model_name="basic"):
        """
        Create CNN models with different architectures
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
                layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
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
                layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
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
                layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
            ])
            
        return model
    
    def compile_model(self, model, num_classes, learning_rate=0.001):
        """
        Compile model with appropriate loss function and metrics
        """
        if num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
            
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        return model
    
    def train_model(self, model, data, model_name, validation_split=0.2):
        """
        Train a model with the given data
        """
        print(f"\nTraining {model_name} model...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                f'{model_name}_best_model.h5',
                monitor='val_accuracy', save_best_only=True
            )
        ]
        
        # Train the model
        history = model.fit(
            data['x_train'], data['y_train'],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, data, model_name):
        """
        Evaluate model performance with comprehensive metrics
        """
        print(f"\nEvaluating {model_name} model...")
        
        # Test evaluation
        eval_results = model.evaluate(data['x_test'], data['y_test'], verbose=0)
        test_loss = eval_results[0]
        test_accuracy = eval_results[1]
        
        # Predictions
        predictions = model.predict(data['x_test'])
        if data['num_classes'] == 2:
            y_pred = (predictions > 0.5).astype(int).flatten()
            y_pred_proba = predictions.flatten()
        else:
            y_pred = np.argmax(predictions, axis=1)
            y_pred_proba = predictions
        
        # Calculate comprehensive metrics using sklearn
        sklearn_accuracy = accuracy_score(data['y_test'], y_pred)
        precision = precision_score(data['y_test'], y_pred, average='weighted', zero_division=0)
        recall = recall_score(data['y_test'], y_pred, average='weighted', zero_division=0)
        f1 = f1_score(data['y_test'], y_pred, average='weighted', zero_division=0)
        
        # AUC Score
        try:
            if data['num_classes'] == 2:
                auc_score = roc_auc_score(data['y_test'], y_pred_proba)
            else:
                auc_score = roc_auc_score(data['y_test'], y_pred_proba, multi_class='ovr', average='weighted')
        except ValueError:
            auc_score = 0.0
        
        # Calculate specificity and sensitivity
        if data['num_classes'] == 2:
            # Binary classification
            tn, fp, fn, tp = confusion_matrix(data['y_test'], y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            # Multi-class classification
            cm = confusion_matrix(data['y_test'], y_pred)
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
            
            specificity = np.mean(specificities)
            sensitivity = np.mean(sensitivities)
            
        # Classification report
        print(f"\n{model_name} Results:")
        print(f"Test Accuracy: {test_accuracy:.4f} (TF) / {sklearn_accuracy:.4f} (sklearn)")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test AUC Score: {auc_score:.4f}")
        print(f"Test Specificity: {specificity:.4f}")
        print(f"Test Sensitivity: {sensitivity:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("\nClassification Report:")
        print(classification_report(data['y_test'], y_pred))
        
        return {
            'test_accuracy': sklearn_accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1_score': f1,
            'test_auc_score': auc_score,
            'test_specificity': specificity,
            'test_sensitivity': sensitivity,
            'test_loss': test_loss,
            'predictions': y_pred,
            'true_labels': data['y_test']
        }
    
    def plot_training_history(self, history, model_name):
        """
        Plot training history
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} Training History', fontsize=16)
        
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
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, results, model_name, class_names=None):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_centralized_training(self):
        """
        Main function to run centralized training on both datasets
        """
        print("Starting Centralized Training with CBIS-DDSM and MIAS datasets")
        print("=" * 70)
        
        # Load datasets
        self.load_and_combine_datasets()
        
        # Training configurations
        configs = [
            {
                'name': 'CBIS_DDSM_Basic',
                'data': self.cbis_data,
                'model_type': 'basic',
                'class_names': ['Benign', 'Malignant']
            },
            {
                'name': 'CBIS_DDSM_Advanced',
                'data': self.cbis_data,
                'model_type': 'advanced',
                'class_names': ['Benign', 'Malignant']
            },
            {
                'name': 'MIAS_Basic',
                'data': self.mias_data,
                'model_type': 'basic',
                'class_names': ['Normal', 'Benign', 'Malignant']
            },
            {
                'name': 'MIAS_Advanced',
                'data': self.mias_data,
                'model_type': 'advanced',
                'class_names': ['Normal', 'Benign', 'Malignant']
            },
            {
                'name': 'Combined_Binary',
                'data': self.combined_data,
                'model_type': 'advanced',
                'class_names': ['Non-Malignant', 'Malignant']
            }
        ]
        
        # Train all models
        all_results = {}
        
        for config in configs:
            print(f"\n{'='*50}")
            print(f"Training: {config['name']}")
            print(f"{'='*50}")
            
            # Create and compile model
            input_shape = config['data']['x_train'].shape[1:]
            num_classes = config['data']['num_classes']
            
            model = self.create_cnn_model(input_shape, num_classes, config['model_type'])
            model = self.compile_model(model, num_classes)
            
            print(f"Model architecture for {config['name']}:")
            model.summary()
            
            # Train model
            history = self.train_model(model, config['data'], config['name'])
            
            # Evaluate model
            results = self.evaluate_model(model, config['data'], config['name'])
            
            # Store results
            all_results[config['name']] = {
                'model': model,
                'history': history,
                'results': results,
                'config': config
            }
            
            # Plot results
            self.plot_training_history(history, config['name'])
            self.plot_confusion_matrix(results, config['name'], config['class_names'])
            
            # Save model
            model.save(f"{config['name']}_final_model.h5")
            print(f"Model saved as {config['name']}_final_model.h5")
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def generate_comparison_report(self, all_results):
        """
        Generate a comparison report of all models
        """
        print("\n" + "="*70)
        print("CENTRALIZED TRAINING COMPARISON REPORT")
        print("="*70)
        
        comparison_data = []
        for name, result in all_results.items():
            comparison_data.append({
                'Model': name,
                'Test Accuracy': f"{result['results']['test_accuracy']:.4f}",
                'Test Precision': f"{result['results']['test_precision']:.4f}",
                'Test Recall': f"{result['results']['test_recall']:.4f}",
                'Test F1 Score': f"{result['results']['test_f1_score']:.4f}",
                'Test AUC Score': f"{result['results']['test_auc_score']:.4f}",
                'Test Specificity': f"{result['results']['test_specificity']:.4f}",
                'Test Sensitivity': f"{result['results']['test_sensitivity']:.4f}",
                'Test Loss': f"{result['results']['test_loss']:.4f}",
                'Architecture': result['config']['model_type']
            })
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)
        print("\nModel Performance Comparison:")
        print(df.to_string(index=False))
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'centralized_training_results_{timestamp}.csv', index=False)
        print(f"\nResults saved to centralized_training_results_{timestamp}.csv")

# Main execution
if __name__ == "__main__":
    # Create trainer instance
    trainer = CentralizedTrainer(
        image_size=(224, 224),
        batch_size=16,  # Adjust based on GPU memory
        epochs=30
    )
    
    # Run centralized training
    results = trainer.run_centralized_training()
    
    print("\nCentralized training completed successfully!")
    print("Check the generated plots and saved models for detailed results.")