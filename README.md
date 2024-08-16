# DQN-based AI for Paper.io (Local Simulation)

This project implements a Deep Q-Network (DQN) based AI agent designed to play a locally simulated version of the game Paper.io. The project was developed during a summer internship at Peking University.

## Project Structure

- **`data.py`**: Handles data preprocessing and preparation for training the DQN model. It processes the game states from the local simulation, and performs operations like tensor rotation and padding to prepare inputs for the neural network.
- **`densenet.py`**: Implements the DenseNet architecture that is used as a feature extractor within the DQN model.
- **`resnet.py`**: Implements the ResNet architecture, which is also used for feature extraction.
- **`net.py`**: Defines the overall DQN architecture which includes components like ResNet and DenseNet.
- **`train.py`**: The main training script that sets up the DQN model, the replay memory, and the training loop. The script interacts with the local simulation of the game environment to optimize the AI's performance.


## Requirements

To run this project, you'll need the following:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- Optionally, a compatible environment for GPU usage.
- This project also requires a local simulation of the game environment, which is provided by a [repository](https://github.com/chbpku/paper.io.sessdsa).

### Usage

1. Clone the local simulation repository:

    ```bash
    git clone https://github.com/chbpku/paper.io.sessdsa
    ```

2. Ensure that the cloned repository is accessible by the project. The `data.py` script imports modules from this repository to simulate the Paper.io environment.

3. Install the necessary dependencies for both the local simulation and this DQN project.

4. Run the `train.py` script to start training the DQN model. The script initializes the local simulation environment, sets up the model, and begins the training process.

    ```bash
    python train.py
    ```