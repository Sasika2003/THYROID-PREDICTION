import pandas as pd
import numpy as np
import tenseal as ts
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import base64

### **🔹 Step 1: Create Encryption Context**
def create_context():
    """Create and configure TenSEAL encryption context"""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 21, 21, 40]
    )
    context.global_scale = 2**21
    context.generate_galois_keys()
    return context

### **🔹 Step 2: Encrypt Data**
def encrypt_data(data, context):
    """Encrypts numerical data using TenSEAL CKKS encryption"""
    print("Encrypting data...")

    # Normalize the data before encryption
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    
    encrypted_data = []
    for row in normalized_data:
        encrypted_row = ts.ckks_vector(context, row.tolist())
        encrypted_data.append(encrypted_row)

    print(f"Encrypted {len(encrypted_data)} rows.")
    return encrypted_data, scaler

### **🔹 Step 3: Preprocess Data**
def preprocess_data(data_path):
    """Loads and preprocesses the dataset"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Selecting categorical and numerical features
    categorical_cols = ['Gender', 'Goitre']
    numerical_cols = ['Age', 'TSH', 'T3', 'T4', 'TT4', 'FTI']

    print("Processing categorical variables...")
    df['Gender'] = (df['Gender'] == 'Male').astype(int)
    df['Goitre'] = (df['Goitre'] == 'Yes').astype(int)

    print("Processing numerical variables...")
    X = df[numerical_cols + categorical_cols].values

    print("Encoding target variable...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Thyroid Disease Diagnosis'])
    
    return X, y, label_encoder, numerical_cols + categorical_cols  # Added column names to return

### **🔹 NEW: Function to Save Encrypted Data**
def save_encrypted_data(encrypted_data, column_names, filename="encrypted_dataset.csv"):
    """Saves encrypted data to CSV file"""
    print(f"Saving encrypted data to {filename}...")
    
    # Convert encrypted vectors to serialized strings
    serialized_data = []
    for enc_row in encrypted_data:
        # Serialize the encrypted vector and encode in base64
        serialized_row = base64.b64encode(enc_row.serialize()).decode('utf-8')
        serialized_data.append(serialized_row)
    
    # Create DataFrame with encrypted data
    df_encrypted = pd.DataFrame({
        'encrypted_data': serialized_data
    })
    
    # Save to CSV
    df_encrypted.to_csv(filename, index=False)
    print(f"Encrypted data saved to {filename}")

### **🔹 Step 4: Softmax Function for Multi-Class Classification**
def softmax(x):
    """Compute softmax probabilities for each class"""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=0)

### **🔹 Step 5: Train Multi-Class Model on Encrypted Data**
def train_multiclass_model(X_encrypted, y, context, n_classes=3, n_iterations=100, learning_rate=0.01):
    """Trains a multi-class logistic regression model using encrypted data"""
    print("Starting multi-class model training...")

    n_samples = len(X_encrypted)
    n_features = len(X_encrypted[0].decrypt())  

    # Initialize weights for each class
    weights = [ts.ckks_vector(context, np.random.normal(0, 0.01, n_features).tolist()) for _ in range(n_classes)]

    # Encrypt y_train as One-Hot Encoded
    y_one_hot = np.eye(n_classes)[y]  

    for iteration in range(n_iterations):
        for i in range(n_samples):
            # Compute scores for all classes
            z_scores = np.array([X_encrypted[i].dot(w).decrypt()[0] for w in weights])
            preds = softmax(z_scores)

            # Compute error
            y_actual = y_one_hot[i]  
            error = preds - y_actual  

            # Update weights for each class
            for c in range(n_classes):
                gradient = X_encrypted[i] * error[c] * learning_rate
                gradient = ts.ckks_vector(context, gradient.decrypt())
                weights[c] = weights[c] - gradient  

        if iteration % 10 == 0:
            print(f"Completed iteration {iteration}")

    return weights

### **🔹 Step 6: Save Model**
def save_model(weights, context, scaler, label_encoder, filename="full_model.pkl"):
    """Saves the full trained model including weights, encryption context, scaler, and label encoder"""
    print("Saving full model...")

    model_data = {
        'weights': [w.decrypt() for w in weights],
        'encryption_context': context.serialize(),
        'scaler': scaler,
        'label_encoder': label_encoder
    }

    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Full model saved as {filename}")

def main():
    try:
        print("Starting the process...")

        # Create Encryption Context
        print("Creating encryption context...")
        context = create_context()

        # Load & Preprocess Data
        print("Preprocessing data...")
        X, y, label_encoder, feature_names = preprocess_data('dataset.csv')

        # Split Data
        print("Splitting data into train & test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Encrypt Training & Test Data
        print("Encrypting training data...")
        X_train_encrypted, scaler = encrypt_data(X_train, context)

        print("Encrypting test data...")
        X_test_encrypted, _ = encrypt_data(X_test, context)

        # NEW: Save Encrypted Training Data
        save_encrypted_data(X_train_encrypted, feature_names, "encrypted_train_data.csv")
        
        # NEW: Save Encrypted Test Data
        save_encrypted_data(X_test_encrypted, feature_names, "encrypted_test_data.csv")

        # Train Multi-Class Model
        print("Training model on encrypted data...")
        trained_weights = train_multiclass_model(X_train_encrypted, y_train, context)

        # Save Full Model
        save_model(trained_weights, context, scaler, label_encoder)

        # Save Encryption Context
        with open('encryption_context.tenseal', 'wb') as f:
            f.write(context.serialize(save_secret_key=True))
        print("Encryption context saved successfully!")

        print("Process completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

### **🔹 Step 7: Run Main Function**
if __name__ == "__main__":
    main()