# Computer Vision Portfolio

---
## Projects

### 1. Visual Privacy Protection

An AI-driven application developed to enhance visual privacy by anonymizing facial features and obfuscating sensitive Personally Identifiable Information (PII) within both images and video streams. This solution addresses critical data privacy concerns by preventing unauthorized identification and information extraction from visual content.

**Key Technologies:**
*   Deep Learning
*   Natural Language Processing (NLP)
*   OpenCV
*   Tkinter (for GUI)

**Code Repository:** [Visual Privacy Protection](https://github.com/akshay-kamath/Visual-Privacy-protection)

---
### 2. YOLO for Object Detection

**Code Repository:** [YOLO for Object Detection](https://github.com/akshay-kamath/Data-Science-Portfolio/blob/main/Computer%20Vision/YOLO_Object_Detection_from_Scratch.ipynb)


This project features a comprehensive, from scratch implementation of the YOLO (You Only Look Once) object detection model using TensorFlow 2.x. It serves as an in-depth educational resource providing a detailed exploration of YOLO's architecture, custom layer development and its unique multi-component loss function. This implementation demonstrates a foundational understanding of object detection algorithms beyond the application of pre-trained models.

**Project Highlights:**
*   **Ground Up Implementation:** All components, from convolutional blocks to final predictions, are manually constructed without reliance on pre built object detection APIs.
*   **Modular Architecture:** Utilizes a custom Keras `MyModel` class for a clear and object-oriented representation of the YOLO architecture.
*   **Custom Multi-Loss Function:** Implements a specialized loss function that simultaneously optimizes for localization, confidence, and classification, crucial for YOLO's efficiency and accuracy.
*   **Detailed Documentation:** Thoroughly commented code and comprehensive explanations demystify the model's internal workings.

**Technical Details:**
### Model Architecture
The YOLO model is structured for optimal speed and accuracy, consisting of two primary logical components:

1.  **The Backbone Network:** This serves as the feature extractor, employing a series of convolutional and max-pooling layers to progressively downsample the input image. Through these layers, the network learns to identify and encode essential features such as edges, textures, and shapes.

2.  **The Detection Head:** This component processes the features from the backbone to generate final predictions. For each cell in the output feature map, it predicts:
    *   **Bounding Box Coordinates:** The (x, y, w, h) values defining the predicted object's location.
    *   **Objectness Score:** A confidence score indicating the probability that the bounding box contains an object.
    *   **Class Probabilities:** The likelihood that the detected object belongs to a specific class.

### The MyModel Class
The model's architecture is encapsulated within the `MyModel` class, which inherits from `tf.keras.Model`. This design provides a clean, object-oriented structure. The `__init__` method of this class defines the following layers:

*   `Conv2D` layers for feature extraction.
*   `MaxPooling2D` layers for downsampling.
*   `Reshape` to flatten the feature map for subsequent sequential layers.
*   `Bidirectional(GRU)` layers for sequence processing, enabling the model to capture context from both forward and backward directions.
*   `Dense` layers to produce the final output predictions.

The `call` method orchestrates the forward pass, defining the data flow through these layers to generate the model's output.

### Custom YOLO Loss Function
The custom Connectionist Temporal Classification (CTC) loss function is integral to the model's performance. It operates as a multi-task loss, integrating three critical error metrics:

*   **Localization Loss:** Utilizes Mean Squared Error (MSE) to penalize inaccuracies in bounding box predictions, driving the model towards precise object localization.
*   **Confidence Loss:** Employs Binary Cross-Entropy to quantify the model's accuracy in predicting whether a grid cell contains an object.
*   **Classification Loss:** Applies Categorical Cross-Entropy to ensure the model correctly identifies the class of each detected object.

This combined loss function enables the model to develop a comprehensive understanding of objects and their properties.
