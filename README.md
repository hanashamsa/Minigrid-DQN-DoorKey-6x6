# Minigrid-DQN-DoorKey-6x6

This repository contains a Deep Q-Network (DQN) implementation in PyTorch for solving the `MiniGrid-DoorKey-6x6-v0` environment. The agent leverages several enhancements, including Double DQN, Huber Loss for stable training, and an Experience Replay Buffer that incorporates a separate buffer for successful episode transitions to accelerate learning.

https://minigrid.farama.org/environments/minigrid/DoorKeyEnv/

The notebook includes all necessary setup, training, evaluation, and ONNX model export steps, making it suitable for participation in reinforcement learning challenges.

## Features

*   **Deep Q-Network (DQN)**: A classic value-based reinforcement learning algorithm.
*   **Double DQN**: Addresses the overestimation bias of Q-values in standard DQN.
*   **Huber Loss**: A loss function that is less sensitive to outliers than mean squared error.
*   **Experience Replay Buffer**: Stores and samples past transitions for more efficient and stable learning.
*   **Success Replay Buffer**: A dedicated buffer for transitions from successful episodes, prioritized to improve exploration and learning in sparse reward environments.
*   **Minigrid Environment Integration**: Custom wrappers for MiniGrid environments to handle image observations and scaling.
*   **PyTorch Implementation**: All models and training logic are built using PyTorch.
*   **ONNX Export**: Provides functionality to export the trained model into ONNX format for deployment.

## Setup

To run this notebook, you'll need a Google Colab environment with GPU access. The following steps outline the setup process:

1.  **Open in Google Colab**: Click the "Open in Colab" badge (if available) or upload the `.ipynb` file to your Colab environment.
2.  **GPU Runtime**: Ensure you're using a GPU runtime. Go to `Runtime > Change runtime type` and select `GPU` as the hardware accelerator.
3.  **Install Dependencies**: The notebook begins with cells to install all required packages using `apt-get` and `pip`.

    ```bash
    !apt-get update
    !apt-get install -y swig python3-numpy python3-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python3-opengl libboost-all-dev libsdl2-dev
    !pip install gymnasium==1.2.3 gymnasium[box2d] pyvirtualdisplay imageio-ffmpeg moviepy==1.0.3
    !pip install onnx onnx2pytorch==0.4.1
    !pip install opencv-python pyvirtualdisplay
    !pip install minigrid==3.0.0
    !pip install onnxscript # Needed for ONNX export with newer PyTorch
    ```

4.  **Mount Google Drive (Optional)**: If you wish to save model checkpoints to your Google Drive, run the cell that mounts your drive.

    ```python
    from google.colab import drive
    drive.mount('/content/gdrive')
    ```

## Usage

### Training the DQN Agent

The notebook is structured to allow you to run cells sequentially for training.

1.  **Initialize Environment**: The `MinigridDoorKey6x6ImgObs` environment is set as the default training environment.

    ```python
    env = MinigridDoorKey6x6ImgObs()
    ```

2.  **Load/Start Training**: The training loop includes options to load a pre-trained model checkpoint or start training from scratch. Set `load_path` if you have a checkpoint.

    ```python
    # load_path = '/content/gdrive/MyDrive/colab-drive/minigrid-model-2022_03_28-10_01_18.p'
    load_path = '' # Start from scratch
    dqn, dqn_target, timesteps = load_checkpoint(load_path)
    ```

3.  **Run Training Loop**: Execute the main training loop cell. It will print progress updates and periodically save checkpoints.

### Evaluating the Model

After training, you can evaluate the agent's performance:

1.  **Agent Class**: The `Agent` class is provided to wrap your trained DQN model for evaluation.
2.  **Run Episodes**: Execute the evaluation cell to run the agent for a specified number of episodes (`N_EPISODES`) and print the average return.

    ```python
    N_EPISODES = 50
    agent = Agent(model=dqn, device=device)
    scores = []
    for i in range(N_EPISODES):
        seed = np.random.randint(1e7)
        scores.append(run_episode(env, agent, seed=seed))
    print("Average Return:", np.mean(scores))
    ```

### Exporting to ONNX

For deployment, the model needs to be exported to ONNX format. A utility function `save_as_onnx` is provided, along with a `CleanMlpMinigridPolicy` class to ensure the exported model is compatible with server-side evaluations (e.g., by baking the `/10.0` scaling into the first layer's weights).

1.  **Run Export Cell**: Execute the cell that reloads the original PyTorch model, creates a clean version, adjusts weights, and exports it.

    ```python
    # ... (code for CleanMlpMinigridPolicy and weight transfer)
    save_as_onnx(clean_model, test_state, 'submission_model_new.onnx')
    print("Exported clean ONNX model!")
    ```

2.  **Download ONNX File**: Use the `files.download` command to get the `submission_model_new.onnx` file.

    ```python
    from google.colab import files
    files.download('submission_model_new.onnx')
    ```
