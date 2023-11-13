# Text Summarization with Transformers

## Introduction

Text Summarization with Transformers is a machine learning project focused on condensing extensive textual information into shorter, coherent summaries without losing the essence and context of the original content. Utilizing the advanced capabilities of Transformer-based models, specifically the BART (Bidirectional and Auto-Regressive Transformers) model, this project aims to deliver state-of-the-art performance in text summarization tasks.

The significance of this project lies in its application across various domains where quick assimilation of information is crucial, such as news aggregation, report generation, and summarizing research papers or lengthy documents.

This project not only showcases the practical implementation of Transformer models in NLP (Natural Language Processing) but also delves deep into the theoretical underpinnings and mathematical foundations that drive these advanced models, offering a comprehensive understanding of their inner workings.

## Background and Theory

### The Transformer Architecture
Transformers, introduced in the seminal paper "Attention Is All You Need" by Vaswani et al., represent a paradigm shift in natural language processing. The key innovation in Transformers is the attention mechanism, which enables the model to focus on different parts of the input sequence when making predictions, regardless of their position. This mechanism is mathematically represented as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q, K, V$ are queries, keys, and values respectively, and $d_k$ is the dimension of the keys.

The Transformer architecture eschews recurrence and instead relies entirely on this self-attention mechanism to draw global dependencies between input and output, making it significantly more parallelizable and efficient for large-scale NLP tasks.

### BART's Pre-Training and Fine-Tuning
BART (Bidirectional and Auto-Regressive Transformers) extends the Transformer architecture by combining bidirectional encoding (similar to BERT) with autoregressive decoding (similar to GPT). Its pre-training involves corrupting text with an arbitrary noising function and learning to reconstruct the original text. The model is effectively learning a joint probability distribution over sequences of text, which is formulated as:

![](/readme_visuals/Autoregressive_Probability_Formula_for_Conditional_Language_Generation.png)

During fine-tuning for summarization tasks, BART adapts to generate concise and relevant summaries from input text, leveraging its pre-trained knowledge.

### Mathematical Foundations of MLM and NSP
The MLM (Masked Language Modeling) task is central to BART's understanding of language context. It is defined as predicting the probability of a masked token given its context, formulated as:


$$P(\text{token}|\text{context}) = \frac{\exp(\text{contextual representation of token})}{\sum_{\text{all tokens}} \exp(\text{contextual representation of each token})}$$


NSP (Next Sentence Prediction) further enhances BART's capability to understand narrative flow. It is a binary classification task where the model predicts if a sentence logically follows another, enriching its comprehension of text structure.

### ROUGE for Summarization Evaluation
ROUGE scores are a set of metrics for evaluating automatic summarization and machine translation. ROUGE-N, for instance, measures the overlap of N-grams between the generated summary and the reference summary. It is quantitatively expressed as:

![](/readme_visuals/ROUGE-N_score_formula.png)

where $Count_{match}(n-gram)$ is the count of matching N-grams in the generated summary.

This project, by integrating these sophisticated NLP techniques and mathematical concepts, aims to push the boundaries of text summarization, offering a deep dive into the cutting-edge of language processing technology.


## Project Workflow

The Text Summarization with Transformers project follows a structured and sequential workflow, ensuring a systematic approach to building and evaluating the text summarization model. The workflow can be broadly divided into the following stages:

### 1. Environment Setup and Dependency Installation
- Essential libraries and frameworks are set up for the project, including PyTorch for neural network operations and Hugging Face's Transformers for accessing pre-trained models.
- Tools for evaluation and experiment tracking, such as Weights & Biases and the Evaluate library, are integrated.

### 2. Data Preparation
- The dataset comprising news articles and their summaries is loaded and preprocessed.
- Preprocessing steps include tokenization and setting up custom data structures to facilitate model training and evaluation.

### 3. Model Initialization and Configuration
- The BART model is initialized using its pre-trained version from the Hugging Face repository.
- Key hyperparameters for the model, such as batch size, learning rate, and training epochs, are configured.

### 4. Training and Validation Loop
- The model is trained over several epochs, with each epoch consisting of a training and a validation phase.
- Training involves feeding batches of data to the model, performing backpropagation, and updating the model weights.
- Validation is conducted to monitor the model's performance on unseen data and prevent overfitting.

### 5. Model Evaluation and Prediction Generation
- Post-training, the model's ability to generate summaries is evaluated using the evaluation dataset.
- The ROUGE metric is employed to quantitatively assess the quality of the generated summaries against actual summaries.

### 6. Results Compilation and Analysis
- The predictions and actual summaries are compiled for detailed analysis.
- Results are visualized and interpreted to understand the model's strengths and areas for improvement.

### 7. Model Deployment and Sharing
- The trained model, along with its tokenizer configuration, is pushed to the Hugging Face Hub for easy access and deployment.
- This step ensures that the model can be readily used or further improved by the community.

This workflow encapsulates the entire process from setting up the environment to deploying the trained model, ensuring a comprehensive approach to tackling the challenge of text summarization with advanced NLP techniques.

## Installation and Setup

To get started with the Text Summarization with Transformers project, follow these steps to set up your environment and install the necessary dependencies.

### Prerequisites
- Python (version 3.7 or later)
- pip (Python package manager)
- Access to a command-line interface

### Step 1: Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/[YourGitHubUsername]/Text_Summarization_with_Transformers.git
cd Text_Summarization_with_Transformers
```
### Step 2: Create a Virtual Environment
It's recommended to create a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate  # For Unix or MacOS
venv\Scripts\activate  # For Windows
```

### Step 3: Install Dependencies
Install all the required libraries using pip:
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Weights & Biases (Optional)
If you want to use Weights & Biases for experiment tracking:
- Sign up for an account on Weights & Biases.
- Set up your API key as per the instructions provided during the signup process.

### Step 5: Hugging Face Hub Setup (Optional)
For interacting with the Hugging Face Hub:
- Sign up for an account on Hugging Face.
- Configure your authentication token as described in the Hugging Face documentation.

## Usage

This section explains how to run the Text Summarization with Transformers project and generate summaries from the provided dataset.

### Running the Notebook
- Open the project's Jupyter notebook (`BART_transformer_summarization.ipynb`) in Jupyter Lab or Jupyter Notebook.
- Run each cell in the notebook sequentially to experience the entire workflow, from data loading to model evaluation.

### Customizing the Dataset
- To use a different dataset for summarization, replace the `BBCarticles.csv` file with your dataset file in a similar format.
- Adjust the data loading and preprocessing code cells as needed to accommodate the format of your new dataset.

### Fine-Tuning the Model
- Modify the hyperparameters in the notebook (such as batch size, learning rate, and number of epochs) to experiment with different training configurations.
- Observe changes in performance and experiment with different settings to find the optimal configuration for your specific use case.

### Model Evaluation
- Evaluate the model's performance using the provided code for ROUGE score calculation.
- Experiment with different evaluation metrics or datasets to gain a deeper understanding of the model's capabilities and limitations.

### Visualizing Training with Weights & Biases
- If using Weights & Biases, you can monitor the training process in real-time through their web interface.
- Analyze various metrics and logs to track the model's progress and make data-driven decisions.

### Contributing to the Hugging Face Hub
- Push your trained model and tokenizer to the Hugging Face Hub to share with the community.
- Collaborate with others and leverage community feedback to enhance the model.

By following these steps, you can effectively utilize and experiment with the Text Summarization with Transformers project, leveraging its capabilities for your NLP tasks.

## Results and Evaluation

The effectiveness of the Text Summarization with Transformers project is quantitatively assessed using the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric. This section provides an overview of the evaluation process and highlights key results.

### ROUGE Score Analysis
- The model's performance in generating summaries is evaluated using the ROUGE score, which compares the overlap between the generated summaries and the reference (actual) summaries.
- ROUGE metrics, including ROUGE-1, ROUGE-2, and ROUGE-L, are calculated, focusing on unigram, bigram, and longest common subsequence overlaps, respectively.
- A higher ROUGE score indicates better quality of the generated summaries in terms of similarity to the reference summaries.

### Model Performance
- The project details the obtained ROUGE scores, providing insights into the model's ability to capture key information and its coherence in summary generation.
- Variations in performance based on different hyperparameter settings and dataset characteristics are discussed, offering a comprehensive view of the model's capabilities.

### Insights and Observations
- Key insights and observations derived from the training and evaluation phases are shared, including aspects like model convergence, overfitting tendencies, and impact of training dataset size.
- Challenges encountered and solutions applied during the project are discussed, providing valuable lessons for similar NLP tasks.

### Visualizations and Metrics
- Visualizations from Weights & Biases are presented to illustrate training dynamics, such as loss curves and other relevant training metrics.
- These visualizations aid in understanding the model's learning process and in identifying areas for further improvement.

The evaluation results demonstrate the model's proficiency in text summarization and provide a benchmark for future enhancements and experiments in the field of NLP.

