[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11110446&assignment_repo_type=AssignmentRepo)

# Neural Networks and Deep Learning Project: Image Captioning
Welcome to our Image Captioning project! This project utilizes a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to generate descriptive captions for images. We have used the DenseNet201 model for image feature extraction and an LSTM model for generating captions, but the modularity of the code allows for any PyTorch integrated CNN to be used.

## Project Summary
Image captioning is a challenging task that combines computer vision and natural language processing. The goal of our project is to develop a model that can automatically generate meaningful captions for images. The model takes an image as input and produces a relevant and coherent caption describing the contents of the image. Our ideas is based on the paper Show and Tell: A Neural Image Caption Generator(https://arxiv.org/pdf/1411.4555.pdf).

Our project focuses on implementing a CNN-RNN architecture for image captioning. The CNN (ResNet18) is responsible for extracting visual features from the input image, while the RNN (LSTM) generates the corresponding caption based on the extracted features.

## Code structure
Our code follows a structured organization to ensure clarity and maintainability. The project structure is as follows:

- `dataset/`: This directory contains the datasets, weights and .pkl files(such as the features precomputed) 
- `models/`: This directory contains other aproaches that we have tried. This includes GPT2 for example.
- `final_notebook_final.ipynb`: This notebook contains the steps followed in this project. It can be downloaded and run (a detailed description is offered inside the notebook).

## Main functionalities and important takes
- Data Exploration
    - Certain objects or activities are way more common than others. Dogs for instance are quite frequent. Also the colour red.
    - We have created some visualisation to better understand how the samples look like.
- Image Data Preprocessing
    - Images have been resized and normalized.
    - We have also created a feature extraction part where the CNN model can be specified.
- Caption Preprocessing
    - Converted to lowercase, removed and unnecesarry whitespaces or single letter words.
    - Added start and endtokens.
    - Created a Tokenizer to convert the words from captions into unique integers.
- Data Loaders
    - Created the dataloader that prepares image-caption pairs for training an image captioning model. 
- The LSTM model for caption generation
    - The model takes image features and caption input as input. It computes the embedded captions, encodes the image features, and concatenates them. The concatenated features are then passed through the decoder LSTM to generate the output sequence. The output sequence is further processed through the FC layers before producing the final predicted caption.
    - The LSTM network starts generating words after each input thus forming a sentence at the end.
- Wandb for finetuning
    - We have used it hyperparameter tuning. Also found out that lr = 0.01 may actually be quite too high.


<img width="1068" alt="image" src="https://github.com/DCC-UAB/dlnn-project_ia-group_10/assets/72266259/3307909c-903d-431e-8c23-bc5e5d7d91fa">

Interactive plot: https://wandb.ai/dl2023team/dlnn-project_ia-group_10/reports/-23-05-29-23-05-09---Vmlldzo0NDk3ODIy

<img width="1068" alt="image" src="https://github.com/DCC-UAB/dlnn-project_ia-group_10/assets/72266259/7cc619cc-bc25-42cf-8657-4f4242371f8b">


Interactive plot but for validation loss: https://wandb.ai/dl2023team/dlnn-project_ia-group_10/reports/-23-05-29-23-07-25---Vmlldzo0NDk3ODMy


## Config used in the Wandb sweep:

<img width="236" alt="image" src="https://github.com/DCC-UAB/dlnn-project_ia-group_10/assets/72266259/2a9da1db-c06d-4768-abf2-f80b9348279f">


## Final tuned model performance:
![image](https://github.com/DCC-UAB/dlnn-project_ia-group_10/assets/72266259/5e4b8a89-6d41-4d0c-8752-2856e84f26b3)


## Qualitative results:

![image](https://github.com/DCC-UAB/dlnn-project_ia-group_10/assets/72266259/65b1c026-e07c-4d96-a129-fc694b7d5100)

We have also used BLEU metric to find the best caption-prediction pairs. Those are the top 15.

## Contributors
- Put your name here
- Eduard Hogea (eduard.hogea00@e-uvt.ro)
- Júlia Garcia Torné (1630382@uab.cat)

Xarxes Neuronals i Aprenentatge Profund,
Grau de Artificial Intelligence, 
UAB, 2023

