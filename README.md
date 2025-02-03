# Image Caption Generator using CNN-LSTM

This project aims to generate image captions using a deep learning model that combines a Convolutional Neural Network (CNN) for image feature extraction and a Long Short-Term Memory (LSTM) network for sequence generation. The model is trained on the Flickr8K dataset.

# Dataset Description

The Flickr8K dataset consists of 8,000 images, each annotated with five different captions. The dataset is widely used for image captioning tasks.

Each entry in the dataset consists of:

Image: The input image from the dataset.

Caption: A textual description of the image.

# 1. Data Preprocessing

Image Feature Extraction: Pretrained CNN (VGG16) is used to extract image features.

Text Tokenization: Captions are tokenized, and sequences are padded to ensure uniform input size.

Vocabulary Creation: Unique words are mapped to integer values.

Train-Test Split: Data is split into training (80%) and testing (20%) sets.

# 2. Model Architecture

Encoder (CNN): A pretrained VGG16 model extracts visual features from input images.

Decoder (LSTM): A sequence model processes the extracted features along with tokenized captions to generate meaningful text.

Embedding Layer: Converts words into dense vector representations.

Dense Layers: Fully connected layers refine the outputs before the final caption generation.

# 3. Model Training

Loss Function: Categorical cross-entropy is used for optimizing predictions.

Optimizer: Adam optimizer is applied for efficient gradient updates.

Dropout Layers: Regularization techniques are implemented to prevent overfitting.

Training Time: Approximate training time depends on the hardware:

T4 x2 GPU: ~90 hours for full training.

# 4. Model Evaluation

BLEU Score: A metric used to evaluate the quality of generated captions compared to human-written ones.

Sample Predictions: Generated captions are compared with actual captions to assess model performance.

Visualization: Images with their respective generated captions are displayed for qualitative analysis.

# 5. Testing and Final Evaluation

The trained model is used to generate captions for new images. The model is tested on unseen images to ensure generalization capability.

Clone the repository:
```bash
git clone https://github.com/21kNabeelUddin/Image-Caption-Generator/edit/main/README.md
```


Install necessary dependencies: Create a virtual environment and install the required libraries using requirements.txt:
```bash
pip install -r requirements.txt
```

move to directory using
```bash
cd Image-Caption-Generator
```

Run the Jupyter notebook: Open the notebook in Jupyter and execute the code cells step by step.
