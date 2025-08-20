import pandas as pd
import numpy as np
import re
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import coo_matrix

# Text preprocessing function
def preprocess_text(text):
    """Clean and standardize text for modeling."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = ' '.join(text.split()) # Remove extra whitespace
    return text

# PyTorch Dataset for support tickets
class SupportTicketDataset(Dataset):
    """Custom Dataset to handle TF-IDF sparse features for PyTorch."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert sparse feature row to a dense tensor
        feature_tensor = torch.FloatTensor(self.features[idx].toarray().flatten())
        label_tensor = torch.LongTensor([self.labels[idx]])
        return feature_tensor, label_tensor.squeeze()

# Neural network for classification
class TextClassifier(nn.Module):
    """Simple feed-forward neural network for classification."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Ensemble model for soft voting
class EnsembleModel(nn.Module):
    """Averages predictions from multiple models."""
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # Average the outputs from all models
        outputs = [model(x) for model in self.models]
        stacked_outputs = torch.stack(outputs)
        return torch.mean(stacked_outputs, dim=0)

# Train a single model instance
def train_single_model(model, train_loader, epochs, learning_rate, device):
    """Train a single TextClassifier instance."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    
    print("--- Training new model instance ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass, backward pass, and optimization
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Print progress
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    return model

# Evaluate the final ensemble model
def evaluate_model(model, test_loader, device):
    """Get predictions and true labels for the test set."""
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

# Save the complete model pipeline
def save_pytorch_model(ensemble_model, vectorizer, label_map, filename='pytorch_ensemble_model.pth'):
    """Save model, vectorizer, and label map to a file."""
    # Bundle components into a single checkpoint dictionary
    checkpoint = {
        'ensemble_state_dict': {f'model_{i}_state_dict': model.state_dict() for i, model in enumerate(ensemble_model.models)},
        'vectorizer': vectorizer,
        'label_map': label_map,
        'model_params': {
            'input_dim': ensemble_model.models[0].layer1.in_features,
            'hidden_dim': ensemble_model.models[0].layer1.out_features,
            'output_dim': ensemble_model.models[0].layer3.out_features
        }
    }
    torch.save(checkpoint, filename)
    print(f"\nModel saved as '{filename}'")

# Load the complete model pipeline
def load_pytorch_model(filename='pytorch_ensemble_model.pth'):
    """Load model, vectorizer, and label map from a file."""
    checkpoint = torch.load(filename, weights_only=False)
    
    # Reconstruct the models from saved parameters
    params = checkpoint['model_params']
    models = []
    for i in range(len(checkpoint['ensemble_state_dict'])):
        model = TextClassifier(params['input_dim'], params['hidden_dim'], params['output_dim'])
        model.load_state_dict(checkpoint['ensemble_state_dict'][f'model_{i}_state_dict'])
        models.append(model)
        
    ensemble_model = EnsembleModel(models)
    vectorizer = checkpoint['vectorizer']
    label_map = checkpoint['label_map']
    
    print(f"Model '{filename}' loaded successfully.")
    return ensemble_model, vectorizer, label_map

# Predict the category for a single ticket
def predict_ticket_pytorch(ensemble_model, vectorizer, label_map, ticket_text, device):
    """Predict category, confidence, and probabilities for a given text."""
    ensemble_model.to(device)
    ensemble_model.eval()
    
    inv_label_map = {v: k for k, v in label_map.items()}
    
    with torch.no_grad():
        # Preprocess, vectorize, and convert to tensor
        processed_text = preprocess_text(ticket_text)
        vectorized_text = vectorizer.transform([processed_text])
        feature_tensor = torch.FloatTensor(vectorized_text.toarray()).to(device)
        
        # Get prediction and probabilities
        output = ensemble_model(feature_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze()
        confidence, predicted_idx = torch.max(probabilities, 0)
        prediction = inv_label_map[predicted_idx.item()]
        
        # Format probabilities for all classes
        prob_dict = {inv_label_map[i]: prob.item() for i, prob in enumerate(probabilities)}

        return {
            'prediction': prediction,
            'probabilities': prob_dict,
            'confidence': confidence.item()
        }

if __name__ == "__main__":
    try:
        # Configuration
        csv_file = 'data/support_tickets.csv'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data = pd.read_csv(csv_file)
        data['processed_text'] = data['text'].apply(preprocess_text)
        data = data[data['processed_text'].str.len() > 0].dropna()
        
        X = data['processed_text']
        y = data['label']

        # Map string labels to integers
        label_map = {label: i for i, label in enumerate(y.unique())}
        y_int = y.map(label_map)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_int, test_size=0.2, random_state=42, stratify=y_int
        )
        print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

        # Vectorize text using TF-IDF
        print("\nVectorizing text with TF-IDF...")
        vectorizer = TfidfVectorizer(
            stop_words='english', max_features=5000, ngram_range=(1, 2)
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Create PyTorch DataLoaders
        train_dataset = SupportTicketDataset(X_train_tfidf, y_train.values)
        test_dataset = SupportTicketDataset(X_test_tfidf, y_test.values)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Train the ensemble model
        print("\nTraining PyTorch Ensemble Model...")
        INPUT_DIM = X_train_tfidf.shape[1]
        HIDDEN_DIM = 128
        OUTPUT_DIM = len(label_map)
        NUM_MODELS = 4
        
        trained_models = []
        for i in range(NUM_MODELS):
            model = TextClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
            trained_model = train_single_model(model, train_loader, epochs=20, learning_rate=0.001, device=device)
            trained_models.append(trained_model)
        
        ensemble_model = EnsembleModel(trained_models)

        # Evaluate the model
        print("\nEvaluating ensemble model...")
        y_true, y_pred = evaluate_model(ensemble_model, test_loader, device)
        
        # Map integer predictions back to string labels
        inv_label_map = {v: k for k, v in label_map.items()}
        y_true_labels = [inv_label_map[i] for i in y_true]
        y_pred_labels = [inv_label_map[i] for i in y_pred]

        print(f"\nEnsemble Test Accuracy: {accuracy_score(y_true_labels, y_pred_labels):.4f}")
        print("\nEnsemble Classification Report:")
        print(classification_report(y_true_labels, y_pred_labels))
        
        # Run predictions on sample tickets
        print("\n--- Running Predictions on Sample Tickets ---")
        sample_tickets = [
            "My computer crashes frequently and won't boot properly",
            "I was charged twice for my monthly subscription",
            "Can I get a demo of your premium features?",
            "The application keeps returning error 404",
        ]
        
        for ticket in sample_tickets:
            result = predict_ticket_pytorch(ensemble_model, vectorizer, label_map, ticket, device)
            print(f"\nTicket: {ticket}")
            print(f"  -> Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")

        # Save and test the loaded model
        model_path='model/ensemble_model.pth'
        save_pytorch_model(ensemble_model, vectorizer, label_map,model_path)
        loaded_model, loaded_vectorizer, loaded_map = load_pytorch_model(model_path)
        
        print("\n--- Testing loaded model ---")
        test_prediction = predict_ticket_pytorch(
            loaded_model, loaded_vectorizer, loaded_map,"I need to update my payment method urgently", device
        )
        print(f"Prediction: {test_prediction['prediction']} (Confidence: {test_prediction['confidence']:.4f})")
        
        print("\n--- SCRIPT COMPLETED SUCCESSFULLY ---")

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()