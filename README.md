## Finance AI - Amazon SageMaker

## Project Overview

This project demonstrates the use of Amazon SageMaker AutoPilot, a tool that automates the process of training and tuning machine learning models. It is particularly useful in the finance sector, where accurate models for predictions and classifications are crucial for decision-making. The goal of this project is to show how AutoPilot can be used to streamline the model development process, from data input to model deployment.

## Workflow

The steps involved in the project are as follows:

1. **Create an Experiment**:
   - Click on "Create Experiment" in Amazon SageMaker AutoPilot.
   - Provide a name for the experiment.

2. **Data Input**:
   - **S3 Bucket Location**: Enter the location of the training data stored in Amazon S3 (this can be found in the accompanying `.ipynb` notebook).
   - **Target Variable**: Specify the column name for the target variable that you want the model to predict.
   - **Output Path**: Provide the S3 path where the output of the experiment should be stored.

3. **Select Problem Type**:
   - Choose whether the task is a **regression** or **classification** problem.

## Running the Experiment

Once the experiment is created:
- SageMaker AutoPilot will automatically generate a candidate generation notebook and a data exploration notebook. You can view these to gain insights into the dataset and model generation process.
- The experiment may take around **2 hours** to complete, as AutoPilot will run approximately **250 tuning jobs** to identify the best hyperparameters.

## Model Selection and Deployment

Once the job is completed:
1. Select the best-performing model based on the results.
2. Click on **Deploy Model** to deploy the chosen model.

## Deployment Steps

1. Provide a name for the **endpoint**.
2. Choose the appropriate **instance type** and the number of instances for the deployment.
3. Click to **deploy the model**.

## Key Features

- **Automated Data Exploration**: SageMaker AutoPilot provides insights into the dataset, helping to understand the key features and structure of the data.
- **Hyperparameter Tuning**: The system runs multiple tuning jobs to optimize model performance automatically.
- **Model Deployment**: Once the model is selected, it can be deployed directly to a SageMaker endpoint for use in production.

## Technologies Used

- **Amazon SageMaker AutoPilot**: For automating model training, hyperparameter tuning, and deployment.
- **Jupyter Notebooks**: For creating and running the experiment.
- **Amazon S3**: For storing the training data and experiment outputs.

## Acknowledgments

Thanks to Amazon for providing comprehensive documentation and tools to make AI model automation accessible.

