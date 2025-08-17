# Natural Language Processing Projects

This repository contains several NLP projects demonstrating various techniques and applications using Hugging Face Transformers and TensorFlow

---

### Sequential Sentence Classification

Built a Deep learning Model for Sequential Sentence Classification, for Converting “Harder to Read” text into “Easier to Read ” text.

- **Code:** [Sequential Sentence Classification](https://github.com/akshay-kamath/Sequential-sentence-classification)
- **License:** Apache License 2.0

## Sentiment Analysis with BERT and Hugging Face Transformers

This Python script demonstrates how to perform sentiment analysis on the IMDb dataset using BERT-based models from the Hugging Face Transformers library. It covers the entire workflow from data preparation to model training, evaluation, and deployment optimization.

- **Code:** [Sentiment Analysis with BERT and Hugging Face Transformers](https://github.com/akshay-kamath/Data-Science-Portfolio/blob/main/Natural%20Language%20Processing/sentiment_analysis_with_bert.py)


### Features

- **Dataset**: Utilizes the IMDb dataset for sentiment analysis.
- **Models**: Implements and fine-tunes various BERT-based models including:
  - `bert-base-uncased`
  - `roberta-base`
  - `microsoft/xtremedistil-l6-h256-uncased`
- **Frameworks**: Uses TensorFlow and the Hugging Face `transformers` and `datasets` libraries.
- **Workflow**:
  - Data loading and preprocessing.
  - Tokenization using `BertTokenizerFast` and `RobertaTokenizerFast`.
  - Building and compiling models using `TFBertForSequenceClassification` and `TFRobertaForSequenceClassification`.
  - Training and validation of the models.
  - Visualization of training history (loss and accuracy).
  - Testing the model with sample inputs.
- **Optimization**:
  - Conversion of the trained model to the ONNX (Open Neural Network Exchange) format for optimized inference.
  - Quantization of the ONNX model to reduce its size and improve performance.
  - Benchmarking and comparison of the TensorFlow, ONNX, and quantized ONNX models.

### Requirements

To run this script, you need to have Python 3 and the following libraries installed:

- `tensorflow`
- `transformers`
- `datasets`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `opencv-python`
- `gensim`
- `Pillow`
- `tensorflow-datasets`
- `tensorflow-probability`
- `google-colab` (if running in Google Colab)
- `tf2onnx`
- `onnxruntime`
- `onnx`
- `coloredlogs`
- `humanfriendly`

You can install the required libraries using pip:

```bash
pip install transformers datasets tensorflow tf2onnx onnxruntime
```

### Script Structure

The script is divided into the following sections:

1.  **Installation**: Installs the required Python libraries.
2.  **Imports**: Imports the necessary modules and libraries.
3.  **Data Preparation**:
    -   Loads the IMDb dataset.
    -   Prepares the data for different models (BERT, RoBERTa, XtremeDistil).
    -   Tokenizes the text data.
    -   Creates TensorFlow datasets for training and validation.
4.  **Modeling**:
    -   Builds sentiment analysis models using different pre-trained transformers.
5.  **Training**:
    -   Sets up the optimizer and loss function.
    -   Trains the model on the training dataset and validates it on the validation dataset.
    -   Plots the training and validation loss and accuracy.
6.  **Testing**:
    -   Tests the trained model with new text inputs to predict sentiment.
7.  **Conversion to ONNX Format**:
    -   Converts the trained TensorFlow model to the ONNX format.
8.  **Inference and Benchmarking**:
    -   Performs inference using the ONNX model.
    -   Compares the performance of the TensorFlow and ONNX models.
9.  **Quantization with ONNX**:
    -   Applies dynamic quantization to the ONNX model to reduce its size.
    -   Evaluates the accuracy of the quantized model.
10. **Understanding Temperature in Distillation**:
    -   Explains the concept of temperature in the context of model distillation using a softmax function example.

---

## Intent Classification for Customer Service using Hugging Face Transformers

This Python script provides a step-by-step guide to building an intent classification model for a customer service chatbot. It utilizes a powerful pre-trained model from the Hugging Face Transformers library and a real-world customer service dataset.

- **Code:** [Intent Classification for Customer Service](https://github.com/akshay-kamath/Data-Science-Portfolio/blob/main/Natural%20Language%20Processing/intent_classification_for_customer_service.py)


### Features

- **Dataset**: Uses the "Training Dataset for Chatbots/Virtual Assistants" from Kaggle which contains 20,000 user utterances categorized by intent.
- **Model**: Implements the `microsoft/deberta-base` model, a state-of-the-art transformer model for sequence classification.
- **Frameworks**: Built with TensorFlow and leverages the Hugging Face `datasets` and `transformers` libraries for efficient data handling and modeling.
- **End-to-End Workflow**:
  - **Data Acquisition**: Downloads and unzips the dataset directly from Kaggle.
  - **Preprocessing**: Loads the data, maps string labels (intents) to integer IDs, and tokenizes the text utterances using `DebertaTokenizerFast`.
  - **Data Pipeline**: Creates a `tf.data.Dataset` pipeline for efficient training, including shuffling, batching and prefetching.
  - **Train/Validation Split**: Splits the dataset into training and validation sets to evaluate the model's performance.
  - **Modeling**: Builds the intent classification model using `TFDebertaForSequenceClassification` with the appropriate number of labels.
  - **Training**: Fine-tunes the pre-trained DeBERTa model on the customer service dataset.
  - **Evaluation**: Visualizes the model's performance using a confusion matrix to understand its accuracy across different intents.

### Requirements

To run this script, you will need Python 3 and the following libraries:

- `tensorflow`
- `transformers`
- `datasets`
- `kaggle`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`

You can install the primary libraries using pip:

```bash
pip install transformers datasets tensorflow kaggle
```

### File Structure

1.  **Installation**: Installs the `transformers` and `datasets` libraries.
2.  **Imports**: Imports all necessary Python libraries.
3.  **Data Preparation**:
    -   Downloads the dataset from Kaggle.
    -   Loads the CSV file into a Hugging Face `Dataset` object.
    -   Identifies unique intents and creates a mapping from intent names to integer labels.
    -   Applies tokenization to the utterances.
    -   Creates and configures a `tf.data.Dataset` for training.
4.  **Modeling**:
    -   Loads the `TFDebertaForSequenceClassification` model, configured for the number of intents in the dataset.
5.  **Training**:
    -   Sets up the optimizer with a learning rate schedule.
    -   Compiles and fits the model to the training data.
6.  **Evaluation**:
    -   Generates predictions on the validation set.
    -   Computes and displays a confusion matrix to visualize the model's performance on each intent.

---

## Named Entity Recognition (NER) using Hugging Face Transformers

This script provides a comprehensive walkthrough for building and training a Named Entity Recognition (NER) model. It uses the popular CoNLL-2003 dataset and fine-tunes a pre-trained RoBERTa model from the Hugging Face Transformers library using TensorFlow.

- **Code:** [Named Entity Recognition](https://github.com/akshay-kamath/Data-Science-Portfolio/blob/main/Natural%20Language%20Processing/Named_entity_recognition_using_huggingface_transformers.py)


### Features

- **Dataset**: Leverages the standard CoNLL-2003 dataset, a benchmark for NER tasks.
- **Model**: Utilizes the `roberta-base` model, fine-tuned for token classification using `TFRobertaForTokenClassification`.
- **Frameworks**: Built with TensorFlow and integrates seamlessly with the Hugging Face `datasets`, `transformers`, and `evaluate` libraries.
- **Core NER Concepts**:
  - **Token Classification**: Treats NER as a task of classifying each token into predefined entity categories (eg Person, Organization, Location).
  - **Label Alignment**: Implements a crucial preprocessing step to align the NER labels with the tokens generated by the RoBERTa tokenizer which often splits words into subwords. Special tokens (like `[CLS]`, `[SEP]`) are assigned a label of `-100` to be ignored by the loss function.
  - **Data Collator**: Uses `DataCollatorForTokenClassification` to dynamically pad sequences in each batch to the length of the longest sequence ensuring efficient training.
- **Evaluation**:
  - Employs the `seqeval` library, a standard metric for sequence labeling tasks, to calculate precision, recall, F1-score and accuracy for each entity type.
- **Inference**: Demonstrates how to use the fine-tuned model to predict named entities in a new, custom sentence.

### Requirements

To run this script, you need Python 3 and the following libraries:

- `tensorflow`
- `transformers`
- `datasets`
- `evaluate`
- `seqeval`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`

You can install the main dependencies using pip:

```bash
pip install transformers datasets evaluate seqeval
```

### Script Structure

1.  **Installation**: Installs the necessary Python packages.
2.  **Imports**: Imports all required libraries and modules.
3.  **Data Preparation**:
    -   Loads the CoNLL-2003 dataset from the Hugging Face Hub.
    -   Initializes the `RobertaTokenizerFast`.
    -   Defines and applies a function to tokenize the input words and align the `ner_tags` with the new subword tokens.
    -   Creates a `DataCollatorForTokenClassification` for batching.
    -   Converts the processed datasets into `tf.data.Dataset` objects for training and validation.
4.  **Modeling**:
    -   Loads the `TFRobertaForTokenClassification` model, configured with the correct number of NER labels.
5.  **Training**:
    -   Sets up an Adam optimizer with a learning rate scheduler.
    -   Compiles the model and trains it on the training dataset.
    -   Visualizes the training and validation loss over epochs.
6.  **Evaluation**:
    -   Loads the `seqeval` metric.
    -   Generates predictions on the validation set.
    -   Processes the predictions and true labels to compute and display detailed classification metrics (precision, recall, F1).
7.  **Testing**:
    -   Performs inference on a sample sentence to demonstrate the model's NER capabilities.

---

## Neural Machine Translation (English → French) using HuggingFace Transformers

This script demonstrates how to build and train a Neural Machine Translation (NMT) model using the Hugging Face Transformers library with TensorFlow. The project focuses on translating English sentences to French using a pre-trained T5-small model and fine-tuning it on a French-English parallel corpus.

- **Code:** [Neural_machine_translation](https://github.com/akshay-kamath/Data-Science-Portfolio/blob/main/Natural%20Language%20Processing/Neural_machine_translation.py)


### Features

- **Data Preparation:** Downloads and preprocesses the French-English dataset from ManyThings.org.
- **Tokenization:** Utilizes `T5TokenizerFast` for efficient tokenization.
- **Model Loading:** Leverages `TFAutoModelForSeq2SeqLM` to load a pre-trained T5-small model.
- **Training:** Fine-tunes the T5-small model on the prepared dataset.
- **Evaluation:** Evaluates the model's performance using the SacreBLEU metric.
- **Inference:** Demonstrates how to use the trained model for new translations.
- **Comparison:** Includes a section to test the original (un-fine-tuned) T5-small model for comparison.

### Installation

To run this script, you need to install the following Python libraries:

```bash
!pip install transformers datasets evaluate sacrebleu
```

### Dataset

The dataset used for training is a French-English parallel corpus (`fra-eng.zip`) sourced from [ManyThings.org](https://www.manythings.org/anki/). This dataset contains numerous English sentences and their corresponding French translations.

### Model

The core of this project is the **T5-small** model from the Hugging Face Transformers library. T5 (Text-to-Text Transfer Transformer) is a powerful encoder-decoder model designed for various text-to-text tasks including machine translation.

### Training

The T5-small model is fine-tuned for 3 epochs on the preprocessed French-English dataset. The training process involves:
- Preparing the dataset for sequence-to-sequence tasks.
- Configuring an optimizer with a learning rate schedule.
- Training the model using `model.fit()` with a batch size of 64.

### Evaluation

The model's translation quality is evaluated using the **SacreBLEU** metric, a widely used metric for assessing the quality of machine translation. It calculates and displays the BLEU score on the validation set.

---

## Extractive Question Answering using HuggingFace Transformers

This script demonstrates how to implement Extractive Question Answering (QA) using the Hugging Face Transformers library with TensorFlow. The project utilizes a pre trained Longformer model to extract answers from given contexts based on user queries.

- **Code:** [Extractive Question Answering](https://github.com/akshay-kamath/Data-Science-Portfolio/blob/main/Natural%20Language%20Processing/Extractive_question_answering.py)


### Features

- **Data Preparation:** Loads and preprocesses a question-answering dataset.
- **Tokenization:** Employs `LongformerTokenizerFast` for efficient tokenization handling long sequences with `stride` and `overflowing_tokens`.
- **Answer Span Identification:** Custom logic to map character level answer spans to token level start and end positions.
- **Model Loading:** Utilizes `TFLongformerForQuestionAnswering` with a pre-trained model (`allenai/longformer-large-4096-finetuned-triviaqa`).
- **Training:** Fine-tunes the Longformer model for the QA task.
- **Evaluation:** Uses the SQuAD metric from the `evaluate` library to assess model performance.
- **Inference:** Demonstrates how to query the trained model with new questions and contexts to extract answers.

### Installation

To run this script, you need to install the following Python libraries:

```bash
!pip install transformers datasets evaluate
```

### Dataset

The primary dataset used  is `covid_qa_deepset` from the Hugging Face Datasets library. This dataset is structured for question-answering tasks, providing contexts, questions and corresponding answers with their start positions.

### Model

The core of this project is the **Longformer** model, specifically `allenai/longformer-large-4096-finetuned-triviaqa`. Longformer is a transformer model designed to handle long documents efficiently, making it suitable for question-answering tasks where context can be extensive.

### Data Preprocessing

The data preprocessing steps are crucial for preparing the dataset for the Longformer model:
- Questions and contexts are tokenized using `LongformerTokenizerFast`.
- `truncation="only_second"` is used to truncate only the context if the combined length exceeds `MAX_LENGTH`.
- `stride=64` and `return_overflowing_tokens=True` are used to handle very long contexts by splitting them into overlapping chunks.
- `offset_mapping` is used to identify the start and end character positions of tokens, which is then used to find the `start_positions` and `end_positions` of the answers within the tokenized input.

### Training

The Longformer model is fine-tuned on the preprocessed dataset. The training process involves:
- Splitting the tokenized dataset into training and validation sets.
- Compiling the model with an Adam optimizer.
- Training the model using `model.fit()` with a batch size of 32.

### Evaluation

The model's performance is evaluated using the **SQuAD metric** from the `evaluate` library. This metric calculates Exact Match (EM) and F1-score which are standard evaluation metrics for extractive question answering.


---
