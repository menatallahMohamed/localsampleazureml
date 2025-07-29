#!/usr/bin/env python3
"""
Azure ML SDK v2 Sample - Local Compute
This script demonstrates how to use Azure ML SDK v2 with local compute.
"""

import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Job, Environment, BuildContext
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.constants import AssetTypes
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_ml_client():
    """
    Create and return an Azure ML client.
    You'll need to provide your Azure subscription details.
    """
    try:
        # Try DefaultAzureCredential first (works with Azure CLI, managed identity, etc.)
        credential = DefaultAzureCredential()
        
        # Replace these with your actual values
        subscription_id = "your-subscription-id"
        resource_group = "your-resource-group"
        workspace_name = "your-workspace-name"
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        logger.info("Successfully created ML Client")
        return ml_client
        
    except Exception as e:
        logger.error(f"Failed to create ML Client: {e}")
        logger.info("Please ensure you're logged in with 'az login' or provide proper credentials")
        return None

def create_training_script():
    """
    Create a simple training script for demonstration.
    """
    script_content = '''
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
'''
    
    with open("train_script.py", "w") as f:
        f.write(script_content)
    
    logger.info("Created training script: train_script.py")

def run_local_job(ml_client):
    """
    Run a job using local compute.
    """
    if not ml_client:
        logger.error("ML Client not available")
        return
    
    try:
        # Create the training script
        create_training_script()
        
        # Define the job
        job = command(
            inputs={
                "test_size": 0.2,
            },
            code="./",  # Local directory containing the script
            command="python train_script.py --test_size ${{inputs.test_size}} --model_output ${{outputs.model_output}}",
            outputs={"model_output": "./outputs"},
            environment="azureml://registries/azureml/environments/sklearn-1.0/versions/1",
            compute="local",  # This specifies local compute
            display_name="local-training-job",
            description="A sample training job running on local compute",
        )
        
        # Submit the job
        logger.info("Submitting job to run on local compute...")
        submitted_job = ml_client.jobs.create_or_update(job)
        
        logger.info(f"Job submitted successfully!")
        logger.info(f"Job name: {submitted_job.name}")
        logger.info(f"Job status: {submitted_job.status}")
        logger.info(f"Job URL: {submitted_job.studio_url}")
        
        return submitted_job
        
    except Exception as e:
        logger.error(f"Failed to run local job: {e}")
        return None

def run_local_script_directly():
    """
    Alternative: Run the training script directly on local machine without Azure ML job submission.
    This is useful for testing and development.
    """
    logger.info("Running training script directly on local compute...")
    
    # Create the training script
    create_training_script()
    
    # Run the script directly
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, "train_script.py", 
            "--test_size", "0.2", 
            "--model_output", "./local_outputs"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Script executed successfully!")
            logger.info("Output:")
            print(result.stdout)
        else:
            logger.error("Script execution failed!")
            logger.error("Error:")
            print(result.stderr)
            
    except Exception as e:
        logger.error(f"Failed to run script: {e}")

def main():
    """
    Main function demonstrating Azure ML SDK v2 usage with local compute.
    """
    logger.info("Azure ML SDK v2 Local Compute Sample")
    logger.info("=" * 50)
    
    # Option 1: Create ML Client and submit job (requires Azure setup)
    logger.info("\nOption 1: Azure ML Job Submission (requires Azure credentials)")
    ml_client = create_ml_client()
    
    if ml_client:
        try:
            # Test connection
            workspaces = list(ml_client.workspaces.list())
            logger.info(f"Connected to Azure ML workspace successfully")
            
            # Run job on local compute
            job = run_local_job(ml_client)
            
        except Exception as e:
            logger.warning(f"Azure ML connection failed: {e}")
            logger.info("Falling back to local execution...")
            ml_client = None
    
    # Option 2: Run directly on local machine (no Azure required)
    if not ml_client:
        logger.info("\nOption 2: Direct Local Execution (no Azure required)")
        run_local_script_directly()
    
    logger.info("\nSample execution completed!")
    logger.info("\nNext steps:")
    logger.info("1. Update the Azure credentials in the script")
    logger.info("2. Replace sample data with your actual dataset")
    logger.info("3. Modify the training script for your specific use case")

if __name__ == "__main__":
    main()
