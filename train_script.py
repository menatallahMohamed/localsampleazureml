
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--model_output", type=str, help="Path to save the model")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    
    args = parser.parse_args()
    
    # Generate sample data if no data path provided
    if not args.data_path:
        print("Generating sample data...")
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(1000) * 0.1
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        data['target'] = y
    else:
        print(f"Loading data from {args.data_path}")
        data = pd.read_csv(args.data_path)
    
    # Prepare features and target
    feature_columns = [col for col in data.columns if col != 'target']
    X = data[feature_columns]
    y = data['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    # Train the model
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save the model
    if args.model_output:
        os.makedirs(args.model_output, exist_ok=True)
        model_path = os.path.join(args.model_output, "model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
