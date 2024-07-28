# Neural Network from scratch

This project is an implementation of a neural network from scratch using NumPy, demonstrating a comprehensive understanding of the fundamental concepts of machine learning and neural networks. The implementation includes various key components such as:

- Inner, Outer, and Hidden Dense Layers: Custom implementation of dense (fully connected) layers to form the network's structure.
- Activation Functions: Utilization of several activation functions including Linear, Sigmoid, ReLU and Softmax, to introduce non-linearity and handle multi-class classification problems.
- Loss Functions: Incorporation of MeanSquaredError loss, categorical cross-entropy loss and Softmax categorical cross-entropy loss to measure the performance of the classification model.
- Forward and Backpropagation Passes: Detailed implementation of forward and backward passes through the network, essential for training.
- Derivatives and Chain Rule: Utilization of derivatives and the chain rule for calculating gradients and updating weights using gradient descent.
- Optimization Techniques: Implementation of optimization methods such as Stochastic Gradient Descent (SGD) to improve model convergence.
- Regularization Techniques: Usage of L1 and L2 regularization to prevent overfitting and improve model generalization.
- Evaluation Metrics: Implementation of accuracy calculation to evaluate model performance.
- Visualization: Visualization of training data and model predictions to better understand data distribution and model behavior.

## Project Structure:

```
.
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── __empty__
├── notebooks/
│   └── neural.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── data_preprocessing.py
│   ├── layers/
│   │   ├── __init__.py
│   │   └── dense.py
│   ├── activations/
│   │   ├── __init__.py
│   │   ├── relu.py
│   │   └── softmax.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── categorical_crossentropy.py
│   │   ├── softmax_categorical_crossentropy.py
│   │   └── mean_squared_error.py
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── sgd.py
│   │   ├── momentum.py
│   │   ├── adagrad.py
│   │   ├── rmsprop.py
│   │   └── adam.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── accuracy.py
│   │   └── gradients.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── neural_network.py
│   │   └── regression.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── validation.py
│   │   └── evaluation.py
│   ├── regularization/
│   │   ├── __init__.py
│   │   ├── l1.py
│   │   ├── l2.py
│   │   └── dropout.py
│   └── saving/
│       ├── __init__.py
│       ├── save_model.py
│       ├── load_model.py
│       └── parameters.py
└── scripts/
    ├── run_training.py
    └── visualize_results.py
```

## Setup:

1\. Clone the repository:

```bash
git clone https://github.com/AbdallahAbou/Neural_Network_From_Scratch
```

2\. Navigate to the project directory:

```bash
cd Neural_Network_From_Scratch
```

3\. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage:

TBD

## Contributing:

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License:
    
This project is licensed under the MIT License - see the LICENSE file for details.