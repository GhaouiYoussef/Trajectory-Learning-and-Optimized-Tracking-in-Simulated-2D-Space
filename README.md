# NN Predicting a Generated Shuttlecock Trajectory

This repository contains code for predicting shuttlecock trajectories using a neural network. The project includes data generation, visualization, and training of a neural network model to predict the trajectory of a shuttlecock.

## Repository Structure

- `frames_NN/`: Directory containing frames generated during training.
- `Presentaion test technique.pptx`: Presentation file for the technical test.
- `PysicsbasedNN.ipynb`: Jupyter notebook containing the code for data generation, visualization, and neural network training.
- `Test - Stage n°2.docx`: Document related to the second stage of the test.

## Requirements

- Python 3.x
- Jupyter Notebook
- NumPy
- Matplotlib
- PyTorch
- Imageio

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/GhaouiYoussef/Trajectory-Learning-and-Optimized-Tracking-in-Simulated-2D-Space
    cd Trajectory-Learning-and-Optimized-Tracking-in-Simulated-2D-Space
    ```

2. Install the required packages:
    ```sh
    pip install numpy matplotlib torch imageio
    ```

## Usage

1. Open the Jupyter notebook:
    ```sh
    jupyter notebook PysicsbasedNN.ipynb
    ```

2. Run the cells in the notebook to generate data, visualize trajectories, and train the neural network model.

## Code Overview

### Data Generation

The `GenDataEquations` class is used to generate shuttlecock trajectory data based on initial velocity and launch angle.

### Visualization

The trajectories are visualized using Matplotlib. The code plots the trajectories and combines the data for training.

### Neural Network Model

The `PiNN` class defines the neural network model using PyTorch. The model is trained to predict the shuttlecock trajectory.

### Training

The `train_model` and `train_model_with_video` functions are used to train the model. The training process includes saving frames and generating a video of the training progress.

## Results

The training progress and loss are displayed for each epoch. A video of the training process is saved as [training_video.mp4](http://_vscodecontentref_/1).

## License

This project is licensed under the MIT License.
