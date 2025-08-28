# NeuralNetFromScratch (C++)

A from-scratch neural network in C++ that learns to classify simple 3x3 patterns (Horizontal, Vertical, Diagonal lines). It includes a tiny interactive Windows console UI where you can toggle pixels with your keyboard and have the trained network predict the pattern and its confidence.

This project is intentionally minimal and educational: no external ML libraries, just vectors, loops, and math. It demonstrates forward propagation, backpropagation, gradient computation, and parameter updates with a simple training loop.

## Table of Contents

- Overview
- Demo: Interactive UI
- Requirements
- Quick Start
- Build and Run
- VS Code Setup (optional)
- How It Works
  - Data and Labels
  - Network Architecture
  - Forward Propagation
  - Loss Function
  - Backpropagation and Gradients
  - Parameter Updates
- Hyperparameters and Configuration
- File-by-File Tour
- Extending the Project
- Troubleshooting
- FAQ
- Roadmap
- References

---

## Overview

The network learns from a small, hard-coded dataset of 3x3 binary images representing three classes:

- Diagonal line
- Vertical line
- Horizontal line

After training, you can draw your own 3x3 pattern in the console grid and press a key to see the predicted class and the model’s softmax confidence.

This is designed for clarity over performance. It keeps the math explicit and contained in a few small classes so you can step through the computations.

## Demo: Interactive UI

The app opens a 3x3 grid in your console and lets you move a cursor across cells, toggling each cell on/off.

- Arrow keys: Move the cursor
- Enter: Toggle the current cell on/off
- C: Classify the current 3x3 pattern

The console then prints something like:

```
Your input is 87.3% a Horizontal.
```

Note: The UI uses Windows console APIs via `windows.h`, so this project targets Windows.

## Requirements

- Windows (uses `windows.h` and console APIs)
- A C++ compiler (tested with MinGW `g++`)
- Optionally: VS Code with the C/C++ extension (launch configuration is provided)

## Quick Start

You can compile and run using the provided batch script or the one-liner below.

Batch script (already included):

```
NeuralNetFromScratch\runNeuralNet.bat
```

Manual compile and run (from `NeuralNetFromScratch` directory):

```
g++ -O2 -std=c++17 -o neural neuralNetwork.cpp inputLayerImp.cpp hiddenLayerImp.cpp outputLayerImp.cpp
./neural
```

If you prefer C++11, change the flag to `-std=c++11`.

## Build and Run

From a Developer Command Prompt or a shell with `g++` in `PATH`:

1) Change directory:

```
cd NeuralNetFromScratch
```

2) Build:

```
g++ -O2 -std=c++17 -o neural neuralNetwork.cpp inputLayerImp.cpp hiddenLayerImp.cpp outputLayerImp.cpp
```

3) Run:

```
./neural
```

Tip: If the provided `runNeuralNet.bat` doesn’t work in your environment, use the manual command above. Batch files don’t support `#` comments, so if you edit it, prefer `REM` for comments.

## VS Code Setup (optional)

This repo ships basic VS Code configuration files in `.vscode/`:

- `NeuralNetFromScratch/.vscode/c_cpp_properties.json`: points IntelliSense at the workspace and `C:/MinGW/bin/gcc.exe`.
- `NeuralNetFromScratch/.vscode/settings.json`: config for C/C++ Runner and useful warnings.
- `NeuralNetFromScratch/.vscode/launch.json`: a sample launch profile using `gdb`.

You can adjust paths to match your local MinGW/GDB installation.

## How It Works

At a high level, training loops over the tiny dataset for a number of epochs, computing the forward pass, loss, backpropagating errors, and updating weights and biases via gradient descent. After training converges below a threshold, the app enters interactive mode and predicts your input patterns.

### Data and Labels

- Input: each 3x3 image is flattened to a length-9 vector of `double` values in `{0, 1}`.
- Labels: one-hot vectors of length 3: `[1,0,0]` = Diagonal, `[0,1,0]` = Vertical, `[0,0,1]` = Horizontal.
- Location: Inline in `NeuralNetFromScratch/neuralNetwork.cpp` (see the `trainingImages` and `targetOutputs` vectors near the top of `main`).

### Network Architecture

- Layers: 3 total (Input → Hidden → Output)
- Input size: 9 (the 3x3 grid)
- Hidden layer: configurable count (default 10 neurons)
- Output size: 3 (Diagonal, Vertical, Horizontal)
- Activations: sigmoid for hidden and output layers
- Prediction: softmax over output logits to get probabilities, then `argmax`

Classes:

- `InputLayer` holds the input vector.
- `HiddenLayer` performs forward pass, computes deltas, gradients, and applies updates.
- `OutputLayer` performs forward pass, computes loss, deltas, gradients, and applies updates; also computes softmax predictions.

### Forward Propagation

For each layer `L`, each neuron computes:

```
z_j = sum_i (a_prev_i * w_{j,i}) + b_j
a_j = sigmoid(z_j)
```

The output layer also stores `logits` (pre-sigmoid `z`) so it can compute a softmax for reporting confidence.

### Loss Function

- Loss: Mean Squared Error (MSE) per image summed over outputs.
- Implementation: `OutputLayer::meanSquaredErrorCostPerImage`

Note: For classification, cross-entropy is common, but MSE works fine for this toy example and keeps derivatives simple.

### Backpropagation and Gradients

Output layer deltas:

```
delta_out_i = d(MSE)/da_i * d(sigmoid)/dz = 2*(target_i - a_i) * a_i*(1-a_i)
```

Hidden layer deltas (by propagating error back through the next layer’s weights and applying sigmoid derivative):

```
delta_hidden_j = (sum_i delta_out_i * w_next_{i,j}) * a_j*(1-a_j)
```

Gradients:

```
∂Loss/∂w_{j,i} = delta_j * a_prev_i
∂Loss/∂b_j     = delta_j
```

### Parameter Updates

Gradient descent style update:

```
w_{j,i} := w_{j,i} + learningRate * (∂Loss/∂w_{j,i})
b_j     := b_j     + learningRate * (∂Loss/∂b_j)
```

Note: These updates add the gradient scaled by `learningRate`. If you want the more conventional “subtract the gradient” form, negate the deltas or use a negative learning rate. As written, the math is internally consistent.

## Hyperparameters and Configuration

You can adjust the following in `NeuralNetFromScratch/neuralNetwork.cpp`:

- `learningRate` — line `NeuralNetFromScratch/neuralNetwork.cpp:146`
- `epochs` — line `NeuralNetFromScratch/neuralNetwork.cpp:147`
- `threshold` (training stops when cost falls below this) — line `NeuralNetFromScratch/neuralNetwork.cpp:149`
- `hiddenLayerNeurons` — line `NeuralNetFromScratch/neuralNetwork.cpp:150`
- `classifications` (output size) — line `NeuralNetFromScratch/neuralNetwork.cpp:151`

Other useful references:

- Console output encoding — `SetConsoleOutputCP(CP_UTF8)` at `NeuralNetFromScratch/neuralNetwork.cpp:85`
- Interactive input handler — function start at `NeuralNetFromScratch/neuralNetwork.cpp:38` (`getUserInput`)
- Prediction call and message — see lines `NeuralNetFromScratch/neuralNetwork.cpp:218` and `NeuralNetFromScratch/neuralNetwork.cpp:225`

## File-by-File Tour

- `NeuralNetFromScratch/neuralNetwork.cpp`: Program entry point. Defines training data/labels, training loop, and the interactive UI (grid drawing, key handling, prediction output).
- `NeuralNetFromScratch/InputLayer.h` + `NeuralNetFromScratch/inputLayerImp.cpp`: Minimal input holder that validates and stores the current input vector.
- `NeuralNetFromScratch/hiddenLayer.h` + `NeuralNetFromScratch/hiddenLayerImp.cpp`: Hidden layer math (forward pass, deltas, gradients, updates).
- `NeuralNetFromScratch/outputLayer.h` + `NeuralNetFromScratch/outputLayerImp.cpp`: Output layer math (loss, forward pass with logits, deltas, gradients, updates, prediction via softmax).
- `NeuralNetFromScratch/runNeuralNet.bat`: Simple compile-and-run helper for Windows using `g++`.
- `NeuralNetFromScratch/.vscode/*`: Optional VS Code configuration for IntelliSense, debugger, and warnings.
- `NeuralNetFromScratch/formulas_neuralNet.docx`: Supporting notes with derivations and formulas.

## Extending the Project

Ideas to explore next:

- Add more training patterns (e.g., crosses or corners) and expand `classifications`.
- Switch to cross-entropy loss and softmax outputs trained directly with that loss.
- Try different activations (ReLU, tanh) and compare behavior.
- Add multiple hidden layers or change hidden size.
- Save/load trained weights to/from disk so you don’t have to retrain on startup.
- Seed RNG predictably for reproducible runs.
- Port to a platform-agnostic UI (remove `windows.h`) or add a simple SDL-based grid.
- Add unit tests for the math functions (sigmoid, gradient shapes, etc.).

## Troubleshooting

- “g++ not found”: Ensure MinGW or a similar toolchain is installed and added to `PATH`.
- Batch comments: If editing `runNeuralNet.bat`, use `REM` for comments instead of `#`.
- Console UI characters look odd: The code sets UTF-8 via `SetConsoleOutputCP(CP_UTF8)`. Ensure your console is using a font/encoding that renders box characters; if not, you can replace the printed characters in `drawGrid`.
- Training seems slow: Reduce `epochs`, increase `learningRate` modestly, or relax `threshold`.
- Predictions look random: Increase `epochs` and/or try a smaller `learningRate`. Verify training data/labels still match your 3-class mapping.

## FAQ

Q: Why use MSE instead of cross-entropy?

A: MSE keeps the derivatives straightforward for a didactic example. Cross-entropy is better for classification, but this project focuses on clarity over best practices.

Q: Why sigmoid everywhere?

A: Simplicity. It’s easy to derive and explain. Feel free to experiment with other activations.

Q: Is this Windows-only?

A: Yes, as written. The interactive grid uses `windows.h`. You can remove the UI and run training/prediction on other platforms.

Q: Can I add more classes?

A: Yes. Append new training examples to `trainingImages`, add corresponding one-hot vectors to `targetOutputs`, and update `classifications` to the new class count.

## Roadmap

- Persist and reload weights
- Cross-entropy + softmax training path
- Optional platform-agnostic UI
- Deterministic seeding and reproducible runs
- Unit tests and CI

## References

- Inline derivations are summarized in `NeuralNetFromScratch/formulas_neuralNet.docx`.
- Classic resources: backpropagation tutorials and notes from standard ML texts (Goodfellow et al., Bishop, etc.).

---

If you have questions or ideas, feel free to open issues or adapt the code. Have fun experimenting!
