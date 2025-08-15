# Time Series Analysis Projects

---

# Project Title
Automatic Speech Recognition with TensorFlow

This project demonstrates how to build and train a basic ASR model using TensorFlow, Keras, and the LJSpeech dataset. The model uses a combination of Convolutional Neural Networks (CNNs) and Bidirectional Gated Recurrent Units (Bi-GRUs) to transcribe speech into text.

## Prerequisites
To run this notebook, you will need the following libraries:

*   **TensorFlow:** The primary deep learning framework.
*   **NumPy:** Used for numerical operations.
*   **Matplotlib:** For plotting and data visualization.
*   **Pandas:** For data manipulation and handling the metadata file.

You can install these dependencies using pip:

```bash
pip install tensorflow numpy matplotlib pandas
```

## Dataset
The model is trained on the LJSpeech-1.1 dataset, which consists of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. The dataset includes a `metadata.csv` file that provides the transcription for each audio clip. The notebook automatically downloads and extracts this dataset.

## Model Architecture
The ASR model is a custom Keras model (`MyModel`) that processes audio spectrograms to predict text. The architecture is as follows:

*   **Feature Extraction:** The audio is first converted into a spectrogram using the Short-Time Fourier Transform (STFT). This spectrogram is then processed by a series of three 2D Convolutional Layers with max-pooling to extract relevant features.
*   **Sequence Processing:** The features are reshaped and fed into two Bidirectional GRU layers. These layers are particularly effective for processing sequential data like audio, as they capture dependencies from both the past and future context.
*   **Output Layer:** A final Dense layer with a softmax activation function predicts the probability distribution over the vocabulary for each time step.

## How to Run
1.  Clone this repository or download the notebook file.
2.  Open the notebook in a compatible environment such as Jupyter Notebook or Google Colab.
3.  Run all the cells in the notebook in sequence. The notebook will automatically download the dataset, preprocess the data, build the model and train it.
After training, the final cells will demonstrate how to use the trained model to make predictions on new audio files from the test set.

## Custom Loss Function
The model uses a custom Connectionist Temporal Classification (CTC) loss function. CTC is essential for ASR models because it handles the misalignment between the input audio sequence and the output text sequence. It allows the model to predict the correct text even when the timing of the spoken words doesn't perfectly match the length of the audio features.

---
### Time Series Analysis & Forecasting with Python

This repository showcases a comprehensive collection of projects focused on time series analysis and forecasting. It covers a wide range of techniques, from fundamental statistical methods to more advanced deep learning models. The primary objective is to demonstrate the practical application of these techniques on real-world datasets to uncover trends, seasonality, and make accurate predictions.

**Key Skills & Technologies:**
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Statistical Modeling:** ARIMA, SARIMA
- **Forecasting:** Statsmodels, pmdarima

**Projects Include:**
- **ARIMA Modeling:** In-depth analysis and forecasting using ARIMA models on datasets like Amazon stock prices, CO2 levels, and milk production.
- **Time Series Manipulation:** Techniques for preprocessing and transforming time series data, including handling missing values and data aggregation.
- **Time Series Analysis:** Comprehensive analysis of time series data, including decomposition, trend analysis, and forecasting.
- **Data Visualization:** Effective visualization techniques for understanding time series patterns, trends, seasonality, and anomalies.

- **Code:** [Explore the Code](https://github.com/akshay-kamath/Time-Series-with-Python)
- **License:** GNU Affero General Public License v3.0
