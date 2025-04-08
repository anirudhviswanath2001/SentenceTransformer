
# Sentence Transformer with Multi-Task Learning

This repository provides a **Sentence Transformer** model fine-tuned for **multi-task learning** to handle both **text classification** and **sentiment analysis** tasks. The model uses the pre-trained **all-MiniLM-L6-v2** transformer model from the **Sentence-Transformers** library, which is further extended for multi-task learning. This repository is designed to help developers and AI practitioners implement and experiment with natural language processing (NLP) tasks using pre-trained models.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation Instructions](#installation-instructions)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [Example Data](#example-data)
- [Model Performance](#model-performance)
- [License](#license)
- [Contact](#contact)

## Project Overview

The **Sentence Transformer** model in this repository is designed for multi-task learning, where it can handle two tasks simultaneously:
1. **Product Category Classification**: Classifies product reviews into predefined categories (e.g., Tech, Clothing, Books).
2. **Sentiment Analysis**: Determines the sentiment of a product review (e.g., Positive, Negative, Neutral).

This architecture is useful for applications where both the sentiment of a review and its category are needed for downstream tasks such as customer feedback analysis, product recommendation, and more.

## Key Features

- **Pre-Trained Model**: Built on the highly efficient **`all-MiniLM-L6-v2`** transformer model, which provides good accuracy with fewer parameters and faster inference.
- **Multi-Task Learning**: Simultaneously performs **product category classification** and **sentiment analysis** on the same input data.
- **Scalable and Efficient**: The transformer backbone is lightweight and can be fine-tuned for specific tasks without needing extensive computational resources.
- **Customizable**: Easily extendable for additional classification tasks or integration with other NLP pipelines.
- **Cross-Platform**: Works across different platforms that support Python and PyTorch, including local machines and cloud-based environments like Google Colab, AWS, or Azure.

## Requirements

To run this project, the following Python libraries are required:

- Python 3.7 or higher
- **PyTorch**: For model training and inference
- **Sentence-Transformers**: For sentence embeddings
- **scikit-learn**: For data processing and metrics
- **pandas**: For data handling
- **numpy**: For numerical operations
- **XGBoost**: For additional classification models (if needed)

### Install Dependencies

Install the required libraries by running the following command:

```bash
pip install -r requirements.txt
