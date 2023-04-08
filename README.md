# Neural Chess Bot in Rust

This repository contains the implementation of a neural chess bot in Rust. The bot uses a self-play approach to guide Monte Carlo Tree Search (MCTS) and aims to create a strong chess-playing AI that emulates the playing style of human Grandmasters.

## Overview

The project's primary goal is to use policy distillation to train bots that can act similarly to real human Grandmasters. By refining the neural network's policy using self-play, we hope to achieve a higher level of chess understanding and create AI opponents that are both challenging and enjoyable to play against.

Once the bot's training is complete, we plan to simulate tournaments using top Grandmaster bots. These simulations will help us evaluate the performance and playing style of our AI and provide valuable insights into the development of future versions.

## Key Components

1. **Monte Carlo Tree Search (MCTS)**: The core of the bot's decision-making process, MCTS is used to explore the game tree and find the best moves in a given position.
2. **Neural Network**: A PyTorch-based neural network guides the MCTS process and helps the bot evaluate positions more accurately.
3. **Self-play**: The bot plays games against itself, continually refining its neural network policy and improving its performance.
4. **Policy Distillation**: This technique is used to create more human-like bots by distilling the knowledge gained during self-play.
5. **Tournament Simulations**: After training, the bot competes in simulated tournaments against top Grandmaster bots to evaluate its performance and playing style.

## Future Work

As the project progresses, we plan to explore additional techniques and improvements to enhance the bot's performance and human-like playing style. We will also consider integrating additional tools and resources, such as opening books and endgame tablebases, to further improve the bot's understanding of chess.

## Contributing

If you are interested in contributing to this project or have any questions, please feel free to open an issue or submit a pull request. We welcome feedback and collaboration from the chess and AI community.