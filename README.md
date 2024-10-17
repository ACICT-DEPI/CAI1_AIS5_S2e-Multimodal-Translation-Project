# Multimodal-Translation-Project-DEPI
This repository contains three machine learning projects focused on natural language processing and computer vision tasks.

## Projects

1. **Machine TranslationV1.0**
   - [File: `Machine_Translation_V1.0.ipynb`][(Notebooks/Machine Translation/Machine_Translation_V1.0.ipynb)](https://github.com/OmarMedhatDev/Multimodal-Translation-Project---DEPI/blob/master/Notebooks/Machine%20Translation/Machine_Translation_V1.0.ipynb)
   - Description: A neural machine translation model for English to Arabic translation using the MarianMT architecture.
   - Key Features:
     - Data preprocessing and analysis
     - Model training and fine-tuning
     - Comparison of original and fine-tuned models

2. **Machine TranslationV1.1**
   - [File: `Machine_Translation_V1.1.ipynb`][(Notebooks/Machine Translation/Machine_Translation_V1.1.ipynb)](https://github.com/OmarMedhatDev/Multimodal-Translation-Project---DEPI/blob/master/Notebooks/Machine%20Translation/Machine_TranslationV1.1.ipynb)
   - Description: An improved neural machine translation model for English to Arabic translation using a Seq2Seq architecture with LSTM layers.
   - Key Features:
     - Comprehensive data preprocessing and cleaning
     - LSTM-based encoder-decoder architecture
     - Inference model for translating new sentences

3. **Chatbot**
   - [File: `Chatbot.ipynb`][(Notebooks/Machine Translation/Machine_Translation_V1.1.ipynb)](https://github.com/OmarMedhatDev/Multimodal-Translation-Project---DEPI/blob/master/Notebooks/Chatbot/Chatbot.ipynb)
   - Description: A conversational AI chatbot using various NLP techniques.
   - Key Features:
     - Data preprocessing and exploratory data analysis
     - Random Forest Classifier for intent classification
     - Sequence-to-Sequence model with attention for response generation
     - Fine-tuning of a pre-trained language model (FLAN-T5)

4. **Image to Text Extraction (OCR)**
   - [File: `Image to text Extraction.ipynb`][(Notebooks/Machine Translation/Machine_Translation_V1.1.ipynb)](https://github.com/OmarMedhatDev/Multimodal-Translation-Project---DEPI/blob/master/Notebooks/Image%20to%20text%20extraction/Image%20to%20text%20Extraction.ipynb)
   - Description: An Optical Character Recognition (OCR) system using deep learning and computer vision techniques.
   - Key Features:
     - Custom CNN model for character recognition
     - Image preprocessing and augmentation
     - Post-processing using computer vision techniques for text extraction

## Setup and Dependencies

To run these projects, you'll need to install the following main dependencies:
tensorflow
transformers
torch
pandas
numpy
matplotlib
seaborn
scikit-learn
opencv-python

You can install these dependencies using pip:

## Usage

Each project file can be run independently. Make sure to update file paths and dataset locations as needed.

1. **Machine Translation**:
   - Ensure you have the required dataset (`ara.txt`)
   - Run the script to train and evaluate the translation model

2. **Chatbot**:
   - Ensure you have the required dataset (`dialog.txt`)
   - Run the script to train the chatbot and interact with it

3. **Image to Text Extraction**:
   - Prepare your OCR dataset
   - Run the script to train the OCR model and process images

## Contributing

Contributions to improve these projects are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some improvement'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a new Pull Request
