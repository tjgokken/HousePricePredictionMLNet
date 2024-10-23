import os
import sys
import json
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers

# Force UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train_and_predict_nn_model(size, bedrooms, zip_code) -> dict[str, float]:
    try:
        # Load dataset
        df = pd.read_csv('VirtualTownHouseDataset.csv')
        
        # Preprocessing
        df = pd.get_dummies(df, columns=['ZipCode'])
        
        X = df[['Size', 'Bedrooms'] + [col for col in df.columns if 'ZipCode_' in col]]
        y = df['Price']
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create neural network model
        model = keras.Sequential([
            layers.Input(shape=(X_train_scaled.shape[1],)),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        # Compile and train
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=0)
        
        # Calculate metrics on test set
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Print metrics in same format as other models
        print("=== Neural Network Results ===", flush=True)
        print(f"R^2: {r2}, MAE: {mae:,.2f}", flush=True)
        
        # Prepare prediction input
        zip_columns = [col for col in df.columns if 'ZipCode_' in col]
        sample = [float(size), float(bedrooms)] + [1 if f'ZipCode_{zip_code}' == col else 0 for col in zip_columns]
        sample_df = pd.DataFrame([sample], columns=X.columns)
        sample_scaled = scaler.transform(sample_df)
        
        # Make prediction
        predicted_price = float(model.predict(sample_scaled, verbose=0)[0][0])
        
        result = {
            "predicted_price": predicted_price,
            "r2": r2,
            "mae": mae
        }
        
        # Print as ASCII-only JSON
        print(json.dumps(result, ensure_ascii=True), flush=True)
        return result

    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(json.dumps({"error": error_msg}, ensure_ascii=True), flush=True)
        return {"error": error_msg}

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(json.dumps({"error": "Required arguments: size bedrooms zip_code"}, ensure_ascii=True))
        sys.exit(1)
    
    try:
        size = float(sys.argv[1])
        bedrooms = float(sys.argv[2])
        zip_code = sys.argv[3]
        train_and_predict_nn_model(size, bedrooms, zip_code)
    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(json.dumps({"error": error_msg}, ensure_ascii=True))
        sys.exit(1)