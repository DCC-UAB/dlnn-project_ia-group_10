[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11110446&assignment_repo_type=AssignmentRepo)

# Neural Networks and Deep Learning Project: Image Captioning
Welcome to our Image Captioning project! This project utilizes a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to generate descriptive captions for images. We have used the ResNet18 model for image feature extraction and an LSTM model for generating captions.

## Project Summary
Image captioning is a challenging task that combines computer vision and natural language processing. The goal of our project is to develop a model that can automatically generate meaningful captions for images. The model takes an image as input and produces a relevant and coherent caption describing the contents of the image.

Our project focuses on implementing a CNN-RNN architecture for image captioning. The CNN (ResNet18) is responsible for extracting visual features from the input image, while the RNN (LSTM) generates the corresponding caption based on the extracted features.

## Code structure
Our code follows a structured organization to ensure clarity and maintainability. The project structure is as follows:


- `data/`: This directory contains the datasets.......
- `models/`: This directory contains....
- `utils/`: This directory includes utility functions for data processing, evaluation metrics, etc...
- `train.py`: This script is used to train the image captioning model.
- `evaluate.py`: This script is used to evaluate the trained model on test data.
- `demo.ipynb`: This Jupyter Notebook provides a step-by-step demonstration of the image captioning process.

  we should modify this, i'm just making it look good :)

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```



## Contributors
- names
- names
- Júlia Garcia Torné (1630382@uab.cat)

Xarxes Neuronals i Aprenentatge Profund
Grau de Artificial Intelligence, 
UAB, 2023

<font color="#FF91AF">Neural Networks and Deep Learning Group</font>
Degree in <font color="#FF69B4">Artificial Intelligence</font>
<font color="#FF69B4">UAB, 2023</font>
