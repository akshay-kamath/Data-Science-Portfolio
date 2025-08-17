# MLOps Pipeline for Tabular Regression with Vertex AI

This project implements an MLOps pipeline for a tabular regression task using Vertex AI Pipelines. The pipeline automates the process of training, evaluating and deploying a machine learning model to predict bike sharing demand.

## Project Structure

- `challenge-notebook.ipynb`: A Jupyter notebook that contains the initial data exploration, model training, and deployment steps. This notebook is used to develop the components of the pipeline.
- `build_pipeline.py`: This script defines the MLOps pipeline using the Kubeflow Pipelines (KFP) SDK. It connects the different components of the pipeline in a directed acyclic graph (DAG).
- `run_pipeline.py`: This script submits the compiled pipeline to Vertex AI for execution.
- `config.json`: A configuration file that contains the parameters for the pipeline such as project ID, region, bucket URI and model hyperparameters.
- `components/`: This directory contains the definitions of the individual pipeline components in YAML format. Each component is a self contained step in the pipeline.
  - `download_data.yaml`: Downloads the dataset from a GCS bucket.
  - `preprocess_data.yaml`: Preprocesses the raw data and splits it into training, validation and test sets.
  - `train.yaml`: Trains a RandomForestRegressor model on the training data.
  - `evaluate_model.yaml`: Evaluates the trained model on the test data and checks if the performance meets a predefined threshold.
  - `register_model.yaml`: Registers the model in the Vertex AI Model Registry if the evaluation is successful.
  - `deploy_model.yaml`: Deploys the registered model to a Vertex AI Endpoint for serving.
- `data/`: This directory contains the raw dataset.
- `model.pkl`: A serialized version of the trained model.

## MLOps Pipeline

The MLOps pipeline consists of the following steps:

1.  **Download Data**: The pipeline starts by downloading the bike sharing dataset from a Google Cloud Storage bucket.
2.  **Preprocess Data**: The downloaded data is then preprocessed and split into training, validation and test sets.
3.  **Train Model**: A Random Forest Regressor model is trained on the preprocessed training data.
4.  **Evaluate Model**: The trained model is evaluated on the test set. The R2 score is used as the evaluation metric.
5.  **Conditional Deployment**:
    - If the R2 score is above a certain threshold (defined in `config.json`), the pipeline proceeds to the next steps.
    - Otherwise, the pipeline stops.
6.  **Register Model**: The model is registered in the Vertex AI Model Registry.
7.  **Deploy Model**: The registered model is deployed to a Vertex AI Endpoint, making it available for online predictions.

## Prerequisites

- A Google Cloud Platform (GCP) project with the Vertex AI API enabled.
- A Google Cloud Storage (GCS) bucket.
- Python 3.7 or later.
- The `gcloud` CLI installed and configured.

## How to Run

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/akshay-kamath/Data-Science-Portfolio/tree/main/MLOps/End-to-End%20MLOps%20for%20Bike%20Demand%20Forecasting
    cd https://github.com/akshay-kamath/Data-Science-Portfolio/tree/main/MLOps/End-to-End%20MLOps%20for%20Bike%20Demand%20Forecasting
    ```

2.  **Install dependencies:**
    
3.  **Configure the project:**

    Update the `config.json` file with your GCP project details, GCS bucket URI, and other pipeline parameters.

4.  **Build the pipeline:**

    ```bash
    python build_pipeline.py
    ```

    This will compile the pipeline defined in `build_pipeline.py` into a JSON file (`tabular-data-regression-kfp-cicd-pipeline.json`).

5.  **Run the pipeline:**

    ```bash
    python run_pipeline.py
    ```

    This will submit the compiled pipeline to Vertex AI for execution.

6.  **Monitor the pipeline:**

    You can monitor the progress of the pipeline run in the Vertex AI section of the Google Cloud Console.

## Dataset Details

[1] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.

```
@article{
	year={2013},
	issn={2192-6352},
	journal={Progress in Artificial Intelligence},
	doi={10.1007/s13748-013-0040-3},
	title={Event labeling combining ensemble detectors and background knowledge},
	url={http://dx.doi.org/10.1007/s13748-013-0040-3},
	publisher={Springer Berlin Heidelberg},
	keywords={Event labeling; Event detection; Ensemble learning; Background knowledge},
	author={Fanaee-T, Hadi and Gama, Joao},
	pages={1-15}
}
```

