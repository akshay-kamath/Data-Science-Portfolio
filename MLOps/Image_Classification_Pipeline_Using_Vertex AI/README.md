# Image Classification MLOps Pipeline using Vertex AI

This project demonstrates how to build and deploy an end-to-end MLOps pipeline for image classification using Google Cloud's Vertex AI. The pipeline automates the process of creating a dataset, training an AutoML image classification model and deploying the trained model to an endpoint for serving predictions.

## Project Structure

- `build_and_deploy.py`: This script initializes and runs the Vertex AI pipeline. It's the entry point for the Cloud Build process.
- `train_pipeline.py`: This file defines the MLOps pipeline using the Kubeflow Pipelines (KFP) SDK. It includes the components for creating a dataset, training the model and deploying it.
- `cloudbuild.yaml`: This file contains the configuration for Google Cloud Build, which automates the process of building the Docker image, pushing it to the Artifact Registry and running the pipeline.
- `Dockerfile`: This file defines the Docker image that contains the necessary dependencies and code to run the pipeline.
- `requirements.txt`: This file lists the Python dependencies required for this project.
- `pipeline.json`: This file is the compiled output of the KFP pipeline, which is used by Vertex AI to execute the pipeline.

## MLOps Pipeline

The MLOps pipeline consists of the following steps:

1.  **Create Dataset**: A Vertex AI Image Dataset is created from a CSV file located in a Google Cloud Storage bucket.
2.  **Train Model**: An AutoML image classification model is trained on the created dataset. The dataset is split into training, validation and test sets.
3.  **Create Endpoint**: A Vertex AI Endpoint is created to host the trained model.
4.  **Deploy Model**: The trained model is deployed to the created endpoint, making it available for online predictions.

## Prerequisites

- A Google Cloud Platform (GCP) project with the Vertex AI and Cloud Build APIs enabled.
- A Google Cloud Storage (GCS) bucket.
- Docker installed and configured to push to the Google Artifact Registry.
- Python 3.7 or later.

## How to Run

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/akshay-kamath/Data-Science-Portfolio/tree/main/MLOps/Image_Classification_Pipeline_Using_Vertex%20AI
    cd https://github.com/akshay-kamath/Data-Science-Portfolio/tree/main/MLOps/Image_Classification_Pipeline_Using_Vertex%20AI
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the project:**

    Update the `PROJECT_ID` and `BUCKET` variables in `build_and_deploy.py` with your GCP project ID and GCS bucket name.

4.  **Build and deploy the pipeline using Cloud Build:**

    ```bash
    gcloud builds submit --config cloudbuild.yaml .
    ```

    This command will trigger a Cloud Build job that will:
    - Build the Docker image.
    - Push the image to the Artifact Registry.
    - Run the `build_and_deploy.py` script to compile and submit the Vertex AI pipeline.

5.  **Monitor the pipeline:**

    You can monitor the progress of the pipeline run in the Vertex AI section of the Google Cloud Console.

## Dependencies

The following Python libraries are used in this project:

- `google-cloud-aiplatform`
- `kfp`
- `google-cloud-pipeline-components`
- `google-cloud-storage`

