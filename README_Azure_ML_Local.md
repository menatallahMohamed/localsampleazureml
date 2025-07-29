# Azure ML SDK v2 Local Compute Setup Guide

This guide shows you how to run Azure ML SDK v2 samples on local compute.

## Prerequisites

1. Python 3.8 or higher
2. Azure subscription (optional for local-only execution)
3. Azure CLI (optional, for Azure integration)

## Installation Steps

### 1. Install Required Packages

```bash
pip install azure-ai-ml azure-identity pandas numpy scikit-learn joblib
```

Or install from requirements file:
```bash
pip install -r requirements_azure_ml.txt
```

### 2. Azure Setup (Optional)

If you want to integrate with Azure ML workspace:

1. Install Azure CLI:
   ```bash
   az login
   ```

2. Set up your Azure credentials by creating a `.env` file:
   ```
   AZURE_SUBSCRIPTION_ID=your-subscription-id
   AZURE_RESOURCE_GROUP=your-resource-group
   AZURE_WORKSPACE_NAME=your-workspace-name
   ```

3. Update the credentials in `azure_ml_local_sample.py`

## Running the Sample

### Option 1: Local Execution Only (No Azure Required)

```bash
python azure_ml_local_sample.py
```

This will:
- Generate sample data
- Train a simple ML model
- Save the model locally
- Display performance metrics

### Option 2: Azure ML Integration with Local Compute

1. Update the Azure credentials in the script
2. Run the script to submit jobs to Azure ML that execute on your local machine

## Key Features Demonstrated

1. **Local Compute Configuration**: How to specify local compute in Azure ML jobs
2. **Job Submission**: Creating and submitting ML jobs using Azure ML SDK v2
3. **Environment Management**: Working with Azure ML environments
4. **Data Handling**: Processing data in local compute scenarios
5. **Model Training**: Training and saving models locally

## Files Created

- `azure_ml_local_sample.py` - Main sample script
- `train_script.py` - Generated training script
- `requirements_azure_ml.txt` - Python dependencies
- `azure_config_example.txt` - Configuration template
- `local_outputs/` - Directory containing trained models

## Folder Structure After Running

```
your-project/
├── azure_ml_local_sample.py
├── train_script.py
├── requirements_azure_ml.txt
├── azure_config_example.txt
├── local_outputs/
│   └── model.pkl
└── .env (create this with your Azure credentials)
```

## Next Steps

1. **Customize the Training Script**: Modify `train_script.py` for your specific ML task
2. **Use Your Data**: Replace the sample data generation with your actual dataset
3. **Add More Features**: Implement logging, monitoring, and advanced ML techniques
4. **Azure Integration**: Set up proper Azure credentials for cloud integration

## Common Issues and Solutions

### Import Errors
- Ensure all packages are installed: `pip install -r requirements_azure_ml.txt`
- Check Python version compatibility

### Azure Authentication Issues
- Run `az login` to authenticate with Azure CLI
- Verify your subscription ID and resource group names
- Ensure you have proper permissions in the Azure ML workspace

### Local Compute Issues
- The script falls back to direct local execution if Azure connection fails
- Local execution works without any Azure setup

## Example Output

When running successfully, you should see output similar to:
```
INFO:__main__:Azure ML SDK v2 Local Compute Sample
INFO:__main__:Running training script directly on local compute...
Generating sample data...
Training model...
Model Performance:
MSE: 0.0097
R2 Score: 0.9993
Model saved to ./local_outputs\model.pkl
Training completed successfully!
```

## Additional Resources

- [Azure ML SDK v2 Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure ML Local Compute Guide](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster)
- [Azure ML SDK v2 Samples](https://github.com/Azure/azureml-examples)
