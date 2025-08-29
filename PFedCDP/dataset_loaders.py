import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def load_cbis_ddsm_dataset(data_path="/Users/nazmusshakibshadhin/Downloads/CBIS", image_size=(224, 224)):
    """
    Load CBIS-DDSM dataset for private client data
    Returns: x_train, y_train, x_test, y_test
    """
    try:
        # Load CSV files
        calc_train = pd.read_csv(os.path.join(data_path, "calc_case_description_train_set.csv"))
        calc_test = pd.read_csv(os.path.join(data_path, "calc_case_description_test_set.csv"))
        mass_train = pd.read_csv(os.path.join(data_path, "mass_case_description_train_set.csv"))
        mass_test = pd.read_csv(os.path.join(data_path, "mass_case_description_test_set.csv"))
        
        # Combine calcification and mass data
        train_data = pd.concat([calc_train, mass_train], ignore_index=True)
        test_data = pd.concat([calc_test, mass_test], ignore_index=True)
        
        # Extract pathology labels (Benign/Malignant)
        train_labels = train_data['pathology'].values
        test_labels = test_data['pathology'].values
        
        # Encode labels: Benign=0, Malignant=1
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_labels)
        y_test = label_encoder.transform(test_labels)
        
        # Load and preprocess images
        x_train = load_dicom_images(train_data, data_path, image_size)
        x_test = load_dicom_images(test_data, data_path, image_size)
        
        print(f"CBIS-DDSM Dataset loaded: Train {x_train.shape}, Test {x_test.shape}")
        print(f"Classes: {label_encoder.classes_}")
        
        return x_train, y_train, x_test, y_test
        
    except Exception as e:
        print(f"Error loading CBIS-DDSM dataset: {e}")
        raise

def load_mias_dataset(data_path="/Users/nazmusshakibshadhin/Downloads/MIAS", image_size=(224, 224)):
    """
    Load MIAS dataset for public server data
    Returns: x_train, y_train, x_test, y_test
    """
    try:
        # Load MIAS info CSV
        mias_info = pd.read_csv(os.path.join(data_path, "mias_derived_info.csv"))
        
        # Extract class information (Normal, Benign, Malignant)
        class_mapping = {
            'Normal': 0,
            'Benign': 1, 
            'Malignant': 2
        }
        
        # Map CLASS column to numeric labels
        labels = []
        for _, row in mias_info.iterrows():
            if row['CLASS'] == 'NORM':
                labels.append(0)  # Normal
            elif row['SEVERITY'] == 'Benign':
                labels.append(1)  # Benign
            elif row['SEVERITY'] == 'Malignant':
                labels.append(2)  # Malignant
            else:
                labels.append(1)  # Default to Benign for other cases
        
        # Load mammogram images
        images = load_mias_images(mias_info, data_path, image_size)
        
        # Split into train/test (80/20)
        split_idx = int(0.8 * len(images))
        x_train, x_test = images[:split_idx], images[split_idx:]
        y_train, y_test = np.array(labels[:split_idx]), np.array(labels[split_idx:])
        
        print(f"MIAS Dataset loaded: Train {x_train.shape}, Test {x_test.shape}")
        print(f"Classes: Normal(0), Benign(1), Malignant(2)")
        
        return x_train, y_train, x_test, y_test
        
    except Exception as e:
        print(f"Error loading MIAS dataset: {e}")
        raise

def load_dicom_images(data_df, base_path, image_size):
    """
    Load and preprocess DICOM images from CBIS-DDSM
    """
    images = []
    for _, row in data_df.iterrows():
        try:
            # Construct image path based on your CBIS directory structure
            image_path = os.path.join(base_path, row['image file path'])
            
            # Load DICOM image (you may need pydicom for this)
            # For now, assuming converted to standard image format
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize and normalize
                img = cv2.resize(img, image_size)
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                images.append(img)
            else:
                # Create placeholder if image not found
                placeholder = np.zeros((*image_size, 1), dtype='float32')
                images.append(placeholder)
                
        except Exception as e:
            print(f"Error loading image {row.get('image file path', 'unknown')}: {e}")
            # Add placeholder
            placeholder = np.zeros((*image_size, 1), dtype='float32')
            images.append(placeholder)
    
    return np.array(images)

def load_mias_images(mias_info, base_path, image_size):
    """
    Load and preprocess MIAS mammogram images
    """
    images = []
    for _, row in mias_info.iterrows():
        try:
            # MIAS images are typically in PGM format
            image_filename = f"{row['REFNUM']}.pgm"
            image_path = os.path.join(base_path, image_filename)
            
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize and normalize
                img = cv2.resize(img, image_size)
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                images.append(img)
            else:
                # Create placeholder if image not found
                placeholder = np.zeros((*image_size, 1), dtype='float32')
                images.append(placeholder)
                
        except Exception as e:
            print(f"Error loading MIAS image {row['REFNUM']}: {e}")
            # Add placeholder
            placeholder = np.zeros((*image_size, 1), dtype='float32')
            images.append(placeholder)
    
    return np.array(images)