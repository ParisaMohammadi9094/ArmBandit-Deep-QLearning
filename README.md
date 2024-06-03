Introduction
The Multi-Armed Bandit problem is a classic example in reinforcement learning. The agent has to choose between multiple actions (or bandits), each providing a random reward from a probability distribution specific to that action. The objective is to maximize the total reward over a series of actions.

This project implements a solution using Deep Q-Networks (DQN) and compares it with a baseline random agent.

Setup
To run this project, you'll need Python 3.7 or higher and the following dependencies:

NumPy
PyTorch
Matplotlib
You can install the dependencies using pip:

bash
Copy code
pip install numpy torch matplotlib
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/multi-armed-bandit-dqn.git
cd multi-armed-bandit-dqn
Run the main script to train the DQN agent and compare it with the random agent:

bash
Copy code
python main.py
The script will output the runtime and display a plot comparing the total rewards of the DQN agent and the random agent across episodes.

Results
The following plot shows the performance of the DQN agent compared to the random agent. The DQN agent learns to choose the actions that maximize the reward over time, while the random agent does not improve its performance.


Contributing
Contributions are welcome! Please open an issue to discuss your ideas or create a pull request with your enhancements.

License
This project is licensed under the MIT License. See the LICENSE file for details.
