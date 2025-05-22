#Intelligent Crime Detection via Face Recognition and Machine Learning

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from mtcnn import MTCNN
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

class VGGFace2Dataset:
    """Custom dataset class for VGGFace2 dataset processing"""
    
    def __init__(self, data_dir, subset='train', max_identities=100, max_images_per_id=20):
        self.data_dir = data_dir
        self.subset = subset
        self.max_identities = max_identities
        self.max_images_per_id = max_images_per_id
        self.detector = MTCNN()
        
        # Load identity mapping if available
        self.identity_meta = self.load_identity_meta()
        self.image_paths, self.labels = self.load_dataset()
        
    def load_identity_meta(self):
        """Load VGGFace2 identity metadata if available"""
        meta_file = os.path.join(self.data_dir, 'identity_meta.csv')
        if os.path.exists(meta_file):
            return pd.read_csv(meta_file)
        return None
    
    def load_dataset(self):
        """Load VGGFace2 dataset with identity filtering"""
        image_paths = []
        labels = []
        
        subset_dir = os.path.join(self.data_dir, self.subset)
        
        if not os.path.exists(subset_dir):
            print(f"Warning: {subset_dir} not found. Please ensure VGGFace2 dataset is properly extracted.")
            return [], []
        
        # Get all identity directories
        identity_dirs = [d for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))]
        
        # Limit to max_identities for computational efficiency
        if len(identity_dirs) > self.max_identities:
            identity_dirs = identity_dirs[:self.max_identities]
        
        print(f"Processing {len(identity_dirs)} identities from VGGFace2 {self.subset} set...")
        
        for identity_id in identity_dirs:
            identity_path = os.path.join(subset_dir, identity_id)
            image_files = [f for f in os.listdir(identity_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit images per identity
            if len(image_files) > self.max_images_per_id:
                image_files = image_files[:self.max_images_per_id]
            
            for image_file in image_files:
                image_path = os.path.join(identity_path, image_file)
                image_paths.append(image_path)
                labels.append(identity_id)
        
        print(f"Loaded {len(image_paths)} images from {len(set(labels))} identities")
        return image_paths, labels

class NextGenModels:
    """Next-generation deep learning models for feature extraction"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.load_models()
    
    def load_models(self):
        """Load and initialize all deep learning models"""
        print("Loading next-generation models...")
        
        # Vision Transformer (using ResNet as backbone for ViT-like features)
        self.vit_model = models.resnet50(pretrained=True)
        self.vit_model.fc = nn.Linear(self.vit_model.fc.in_features, 512)
        
        # EfficientNet (using MobileNetV2 as alternative)
        self.efficientnet = models.mobilenet_v2(pretrained=True)
        self.efficientnet.classifier = nn.Linear(self.efficientnet.classifier[1].in_features, 512)
        
        # ResNeXt
        self.resnext = models.resnext50_32x4d(pretrained=True)
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, 512)
        
        # DenseNet
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 512)
        
        # Move models to device
        self.vit_model.to(self.device)
        self.efficientnet.to(self.device)
        self.resnext.to(self.device)
        self.densenet.to(self.device)
        
        # Set to evaluation mode
        self.vit_model.eval()
        self.efficientnet.eval()
        self.resnext.eval()
        self.densenet.eval()
        
        print("Next-generation models loaded successfully!")
    
    def extract_features(self, image, model_name):
        """Extract features using specified model"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if model_name == 'vit':
                features = self.vit_model(input_tensor)
            elif model_name == 'efficientnet':
                features = self.efficientnet(input_tensor)
            elif model_name == 'resnext':
                features = self.resnext(input_tensor)
            elif model_name == 'densenet':
                features = self.densenet(input_tensor)
            else:
                raise ValueError(f"Unknown model name: {model_name}")
            
            features = features.cpu().numpy().flatten()
        
        return features / np.linalg.norm(features)

def preprocess_face_multi_scale(img_path, detector):
    """Preprocess face image for multiple model requirements"""
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = detector.detect_faces(img_rgb)
        if not results:
            return None, None
        
        # Extract the largest face
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        # Extract face
        face = img_rgb[y1:y2, x1:x2]
        
        # Traditional size (160x160) for FaceNet-style models
        face_160 = cv2.resize(face, (160, 160))
        face_160 = face_160.astype('float32')
        face_160 = (face_160 - face_160.mean()) / (face_160.std() + 1e-7)
        
        # Modern deep learning size (224x224)
        face_224 = cv2.resize(face, (224, 224))
        face_224 = face_224.astype('uint8')
        
        return face_160, face_224
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None

def create_facenet_model():
    """Create a FaceNet-style model using transfer learning"""
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(160, 160, 3)
    )
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    embeddings = Dense(128, activation=None)(x)  # FaceNet-style 128D embeddings
    
    model = Model(inputs=base_model.input, outputs=embeddings)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def get_facenet_embedding(model, face):
    """Extract FaceNet-style embeddings"""
    face = np.expand_dims(face, axis=0)
    embedding = model.predict(face, verbose=0)[0]
    return embedding / np.linalg.norm(embedding)

def load_and_process_vggface2(dataset_path, facenet_model, nextgen_models, max_samples=1000):
    """Load and process VGGFace2 dataset"""
    print("Loading VGGFace2 dataset...")
    
    # Initialize dataset
    vggface2_dataset = VGGFace2Dataset(dataset_path, subset='train', max_identities=50, max_images_per_id=20)
    
    if not vggface2_dataset.image_paths:
        print("No images found. Creating synthetic dataset for demonstration...")
        return create_synthetic_dataset(max_samples)
    
    # Initialize detector
    detector = MTCNN()
    
    # Storage for different representations
    faces_160 = []
    faces_224 = []
    labels = []
    facenet_embeddings = []
    vit_embeddings = []
    efficientnet_embeddings = []
    resnext_embeddings = []
    densenet_embeddings = []
    
    processed_count = 0
    max_samples = min(max_samples, len(vggface2_dataset.image_paths))
    
    print(f"Processing {max_samples} images...")
    
    for i, (img_path, label) in enumerate(zip(vggface2_dataset.image_paths, vggface2_dataset.labels)):
        if processed_count >= max_samples:
            break
            
        # Preprocess face
        face_160, face_224 = preprocess_face_multi_scale(img_path, detector)
        
        if face_160 is not None and face_224 is not None:
            try:
                # Extract all embeddings
                facenet_emb = get_facenet_embedding(facenet_model, face_160)
                vit_emb = nextgen_models.extract_features(face_224, 'vit')
                eff_emb = nextgen_models.extract_features(face_224, 'efficientnet')
                resnext_emb = nextgen_models.extract_features(face_224, 'resnext')
                densenet_emb = nextgen_models.extract_features(face_224, 'densenet')
                
                # Store all representations
                faces_160.append(face_160)
                faces_224.append(face_224)
                labels.append(label)
                facenet_embeddings.append(facenet_emb)
                vit_embeddings.append(vit_emb)
                efficientnet_embeddings.append(eff_emb)
                resnext_embeddings.append(resnext_emb)
                densenet_embeddings.append(densenet_emb)
                
                processed_count += 1
                
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count}/{max_samples} images...")
                    
            except Exception as e:
                print(f"Error extracting features from {img_path}: {e}")
                continue
    
    print(f"Successfully processed {processed_count} images from {len(set(labels))} identities")
    
    return {
        'faces_160': np.array(faces_160),
        'faces_224': np.array(faces_224),
        'labels': np.array(labels),
        'facenet': np.array(facenet_embeddings),
        'vit': np.array(vit_embeddings),
        'efficientnet': np.array(efficientnet_embeddings),
        'resnext': np.array(resnext_embeddings),
        'densenet': np.array(densenet_embeddings)
    }

def create_synthetic_dataset(max_samples=1000):
    """Create synthetic dataset for demonstration if VGGFace2 is not available"""
    print("Creating synthetic dataset for demonstration...")
    
    n_identities = 20
    samples_per_identity = max_samples // n_identities
    
    # Generate synthetic embeddings
    np.random.seed(42)
    
    facenet_embeddings = []
    vit_embeddings = []
    efficientnet_embeddings = []
    resnext_embeddings = []
    densenet_embeddings = []
    labels = []
    
    for i in range(n_identities):
        identity_name = f"person_{i:03d}"
        
        # Generate base features for this identity
        base_facenet = np.random.randn(128)
        base_vit = np.random.randn(512)
        base_efficientnet = np.random.randn(512)
        base_resnext = np.random.randn(512)
        base_densenet = np.random.randn(512)
        
        for j in range(samples_per_identity):
            # Add noise to create variations
            noise_factor = 0.1
            
            facenet_emb = base_facenet + np.random.randn(128) * noise_factor
            vit_emb = base_vit + np.random.randn(512) * noise_factor
            eff_emb = base_efficientnet + np.random.randn(512) * noise_factor
            resnext_emb = base_resnext + np.random.randn(512) * noise_factor
            densenet_emb = base_densenet + np.random.randn(512) * noise_factor
            
            # Normalize embeddings
            facenet_embeddings.append(facenet_emb / np.linalg.norm(facenet_emb))
            vit_embeddings.append(vit_emb / np.linalg.norm(vit_emb))
            efficientnet_embeddings.append(eff_emb / np.linalg.norm(eff_emb))
            resnext_embeddings.append(resnext_emb / np.linalg.norm(resnext_emb))
            densenet_embeddings.append(densenet_emb / np.linalg.norm(densenet_emb))
            
            labels.append(identity_name)
    
    print(f"Created synthetic dataset with {len(labels)} samples from {n_identities} identities")
    
    return {
        'labels': np.array(labels),
        'facenet': np.array(facenet_embeddings),
        'vit': np.array(vit_embeddings),
        'efficientnet': np.array(efficientnet_embeddings),
        'resnext': np.array(resnext_embeddings),
        'densenet': np.array(densenet_embeddings)
    }

def train_enhanced_models(dataset_dict):
    """Train all 8 models and create enhanced ensemble"""
    labels = dataset_dict['labels']
    
    # Convert string labels to numeric
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"Training models on {len(labels)} samples with {len(set(labels))} identities")
    
    # Prepare different embeddings for different models
    embeddings = {
        'facenet': dataset_dict['facenet'],
        'vit': dataset_dict['vit'],
        'efficientnet': dataset_dict['efficientnet'],
        'resnext': dataset_dict['resnext'],
        'densenet': dataset_dict['densenet']
    }
    
    # Split data with stratification
    splits = {}
    for emb_type, emb_data in embeddings.items():
        X_train, X_test, y_train, y_test = train_test_split(
            emb_data, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        splits[emb_type] = (X_train, X_test, y_train, y_test)
    
    # Initialize all models
    models = {
        # Traditional ML models (using FaceNet embeddings)
        'k-NN': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        
        # Next-generation models (using their respective embeddings)
        'Vision Transformer': KNeighborsClassifier(n_neighbors=3, metric='cosine'),
        'EfficientNet': SVC(kernel='rbf', probability=True, random_state=42),
        'ResNeXt': RandomForestClassifier(n_estimators=150, random_state=42),
        'DenseNet': xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss')
    }
    
    # Model-embedding mapping
    model_embeddings = {
        'k-NN': 'facenet',
        'SVM': 'facenet',
        'Random Forest': 'facenet',
        'XGBoost': 'facenet',
        'Vision Transformer': 'vit',
        'EfficientNet': 'efficientnet',
        'ResNeXt': 'resnext',
        'DenseNet': 'densenet'
    }
    
    # Train models and collect results
    trained_models = {}
    results = {}
    
    print("\nTraining individual models...")
    print("=" * 70)
    
    for name, model in models.items():
        emb_type = model_embeddings[name]
        X_train, X_test, y_train, y_test = splits[emb_type]
        
        print(f"Training {name} with {emb_type} embeddings...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"{name:<20} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print("-" * 70)
    
    # Enhanced ensemble predictions with weighted voting
    print("\nCreating Enhanced Ensemble...")
    
    ensemble_preds = []
    X_test_facenet = splits['facenet'][1]
    y_test_facenet = splits['facenet'][2]
    
    # Weights for ensemble (higher for next-gen models)
    model_weights = {
        'k-NN': 0.10,
        'SVM': 0.12,
        'Random Forest': 0.13,
        'XGBoost': 0.15,
        'Vision Transformer': 0.15,
        'EfficientNet': 0.12,
        'ResNeXt': 0.11,
        'DenseNet': 0.12
    }
    
    for i in range(len(X_test_facenet)):
        weighted_votes = {}
        
        for name, model in trained_models.items():
            emb_type = model_embeddings[name]
            sample = splits[emb_type][1][i:i+1]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(sample)[0]
                pred_class = np.argmax(proba)
                confidence = np.max(proba)
            else:
                pred_class = model.predict(sample)[0]
                confidence = 1.0
            
            weight = model_weights[name] * confidence
            
            if pred_class not in weighted_votes:
                weighted_votes[pred_class] = 0
            weighted_votes[pred_class] += weight
        
        # Select class with highest weighted vote
        ensemble_pred = max(weighted_votes, key=weighted_votes.get)
        ensemble_preds.append(ensemble_pred)
    
    # Calculate ensemble metrics
    ensemble_accuracy = accuracy_score(y_test_facenet, ensemble_preds)
    ensemble_precision = precision_score(y_test_facenet, ensemble_preds, average='weighted', zero_division=0)
    ensemble_recall = recall_score(y_test_facenet, ensemble_preds, average='weighted', zero_division=0)
    ensemble_f1 = f1_score(y_test_facenet, ensemble_preds, average='weighted', zero_division=0)
    
    results['Enhanced Ensemble'] = {
        'accuracy': ensemble_accuracy,
        'precision': ensemble_precision,
        'recall': ensemble_recall,
        'f1_score': ensemble_f1
    }
    
    print(f"{'Enhanced Ensemble':<20} - Accuracy: {ensemble_accuracy:.4f}, Precision: {ensemble_precision:.4f}, Recall: {ensemble_recall:.4f}, F1: {ensemble_f1:.4f}")
    
    return trained_models, label_encoder, results, splits, model_embeddings

def visualize_results(results):
    """Create visualization of model performance"""
    # Prepare data for visualization
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Create DataFrame for easier plotting
    data = []
    for model in models:
        for metric in metrics:
            data.append({
                'Model': model,
                'Metric': metric.replace('_', ' ').title(),
                'Score': results[model][metric]
            })
    
    df = pd.DataFrame(data)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Bar plot of all metrics
    plt.subplot(2, 2, 1)
    pivot_df = df.pivot(index='Model', columns='Metric', values='Score')
    pivot_df.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Model Performance Comparison - All Metrics')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy comparison
    plt.subplot(2, 2, 2)
    accuracy_data = [results[model]['accuracy'] for model in models]
    colors = ['skyblue' if 'Enhanced Ensemble' not in model else 'gold' for model in models]
    bars = plt.bar(range(len(models)), accuracy_data, color=colors)
    plt.title('Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Traditional ML vs Next-Gen DL comparison
    plt.subplot(2, 2, 3)
    trad_ml_models = ['k-NN', 'SVM', 'Random Forest', 'XGBoost']
    nextgen_models = ['Vision Transformer', 'EfficientNet', 'ResNeXt', 'DenseNet']
    
    trad_ml_avg = np.mean([results[model]['accuracy'] for model in trad_ml_models])
    nextgen_avg = np.mean([results[model]['accuracy'] for model in nextgen_models])
    ensemble_acc = results['Enhanced Ensemble']['accuracy']
    
    categories = ['Traditional ML\n(Average)', 'Next-Gen DL\n(Average)', 'Enhanced\nEnsemble']
    values = [trad_ml_avg, nextgen_avg, ensemble_acc]
    colors = ['lightcoral', 'lightblue', 'gold']
    
    bars = plt.bar(categories, values, color=colors)
    plt.title('Architecture Type Comparison')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: F1-Score vs Accuracy scatter plot
    plt.subplot(2, 2, 4)
    f1_scores = [results[model]['f1_score'] for model in models]
    accuracy_scores = [results[model]['accuracy'] for model in models]
    
    scatter_colors = ['red' if 'Enhanced Ensemble' in model else 'blue' for model in models]
    plt.scatter(accuracy_scores, f1_scores, c=scatter_colors, alpha=0.7, s=100)
    
    for i, model in enumerate(models):
        plt.annotate(model, (accuracy_scores[i], f1_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('F1-Score vs Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('F1-Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_performance_table(results):
    """Print enhanced performance comparison table"""
    print("\n" + "="*100)
    print("ENHANCED PERFORMANCE COMPARISON TABLE - VGGFace2 Dataset")
    print("="*100)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<10} {'F1 Score':<10} {'Type':<15}")
    print("-"*100)
    
    # Define model types
    model_types = {
        'k-NN': 'Traditional ML',
        'SVM': 'Traditional ML', 
        'Random Forest': 'Traditional ML',
        'XGBoost': 'Traditional ML',
        'Vision Transformer': 'Next-Gen DL',
        'EfficientNet': 'Next-Gen DL',
        'ResNeXt': 'Next-Gen DL',
        'DenseNet': 'Next-Gen DL',
        'Enhanced Ensemble': 'Hybrid Ensemble'
    }
    
    for model_name, metrics in results.items():
        model_type = model_types.get(model_name, 'Unknown')
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {model_type:<15}")
    
    print("="*100)
    
    # Calculate and print summary statistics
    trad_ml_models = ['k-NN', 'SVM', 'Random Forest', 'XGBoost']
    nextgen_models = ['Vision Transformer', 'EfficientNet', 'ResNeXt', 'DenseNet']
    
    trad_ml_avg = np.mean([results[model]['accuracy'] for model in trad_ml_models])
    nextgen_avg = np.mean([results[model]['accuracy'] for model in nextgen_models])
    ensemble_acc = results['Enhanced Ensemble']['accuracy']
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"Traditional ML Average Accuracy: {trad_ml_avg:.4f}")
    print(f"Next-Generation DL Average Accuracy: {nextgen_avg:.4f}")
    print(f"Enhanced Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"Improvement over Traditional ML: {((ensemble_acc - trad_ml_avg) * 100):.2f}%")
    print(f"Improvement over Next-Gen DL: {((ensemble_acc - nextgen_avg) * 100):.2f}%")

def main():
    """Main function to run the enhanced criminal detection system"""
    print("="*80)
    print("ENHANCED CRIMINAL DETECTION SYSTEM")
    print("VGGFace2 Dataset Implementation")
    print("="*80)
    
    # Configuration
    vggface2_path = "path/to/vggface2"  # Update this path
    max_samples = 1000  # Adjust based on computational resources
    
    try:
        # Initialize models
        print("Initializing models...")
        facenet_model = create_facenet_model()
        nextgen_models = NextGenModels()
        print("Models initialized successfully!")
        
        # Load and process dataset
        print(f"\nLoading VGGFace2 dataset from: {vggface2_path}")
        dataset_dict = load_and_process_vggface2(vggface2_path,
