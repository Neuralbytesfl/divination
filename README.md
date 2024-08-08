Fractal Image Generator
Overview
The Fractal Image Generator is a Python-based project that utilizes Pygame to simulate and visualize movements and interactions in a fractal-like, four-dimensional space projected onto a 2D screen. The primary focus of this project is to explore the dynamics of points in a simulated space as they evolve over time using concepts from reinforcement learning and geometric transformations.

Algorithm Description
Fractal Generation and Dynamics
The algorithm maintains a list of points in a simulated 4D space. Each point can move in this space based on a set of predefined actions. These actions adjust the point's position slightly in one of eight possible directions, influenced by the principles of reinforcement learning, particularly using a Q-learning approach.

Key Components
State Representation: Each point's position in 4D space is treated as a state.
Action Set: Actions are vectors that slightly alter a point's position in 4D space.
Q-Table: A lookup table where each entry corresponds to a state-action pair, holding a value (Q-value) that represents the expected future rewards for taking that action in that state.
Learning Mechanisms: The system updates its Q-values based on the rewards received from moving points, facilitating the learning of more effective movements over time.
Equations
State Update:

s(t+1) = s(t) + a(t)
Where s(t) is the current state, a(t) is the action taken, and s(t+1) is the new state after the action.
Q-Value Update (Q-learning):

Q(s(t), a(t)) = Q(s(t), a(t)) + alpha * [r(t) + gamma * max(Q(s(t+1), a')) - Q(s(t), a(t))]
Here, alpha is the learning rate, gamma is the discount factor, r(t) is the reward received, and max(Q(s(t+1), a')) is the maximum Q-value achievable from the next state.
Projection to 2D:

Points are rotated in 4D space using rotation matrices and then projected onto a 2D screen for visualization.
Rotation and Projection
The rotation in 4D is followed by a projection to 3D and eventually to 2D to display the points on the screen. This involves complex mathematical transformations using cosine and sine functions based on predefined angles that change incrementally to simulate rotation.

Usage
To run the Fractal Image Generator:

Ensure Python and Pygame are installed.
Clone this repository and navigate to the directory containing the script.
Run the script using Python:
bash
Copy code
python fractal_image_generator.py
Dependencies
Python 3.x
Pygame
License
This project is licensed under the MIT License - see the LICENSE file for details.
