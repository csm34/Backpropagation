# Neural Network Implementation with Backpropagation

This code shows a simple feedforward neural network steps with backpropagation using sigmoid activation functions. The network is trained to minimize the Mean Squared Error (MSE) between the expected output and the predicted output.

## Overview

The neural network consists of:
- Input Layer with two input values (`x1` and `x2`).
- Hidden Layer with two neurons (`h1` and `h2`).
- Output Layer with one output neuron (`output1`).

The network uses the Sigmoid Activation Function and performs forward pass and backpropagation to update weights using the Gradient Descent.

## Components

### 1. Sigmoid Activation Function

The sigmoid function is defined as:

$$
Q(x) = \frac{1}{1 + e^{-x}}
$$

It maps any input value to a value between 0 and 1, which is useful for binary classification problems.

### 2. Sigmoid Derivative

The derivative of the sigmoid function is:

$$
Q'(x) = Q(x) \cdot (1 - Q(x))
$$

This derivative is used during backpropagation to compute gradients.

### 3. Forward Propagation

In forward propagation, the following steps occur:
1. Hidden Layer Calculations
2. Output Layer Calculation
   
### 4. Backpropagation

During backpropagation, the following steps are performed:
1. Compute the error using Mean Squared Error (MSE)
2. Compute Derivatives
3. Update Weights
4. After performing forward propagation, backpropagation, and weight updates, the new predicted output and MSE are computed and compared to the previous output to check the improvement in prediction.

### 5. Explanation

- Inputs: `x1 = 0.5`, `x2 = 0.3`, expected output `y_expected = 1`, and learning rate `n = 0.1`.
- Initial Weights:  
  The initial weights are:  
  `w1 = 0.7010`, `w2 = 0.3009`, `w3 = 0.4011`, `w4 = 0.6005`, `w5 = 0.551`, `w6 = 0.4595`.

### 6. Functions

- `sigmoid(x)`: Returns the sigmoid of input `x`.
- `sigmoid_derivative(x)`: Returns the derivative of the sigmoid of input `x`.
- `l1f1(x1, x2)` and `l1f2(x1, x2)`: Compute the activations of the two hidden neurons.
- `of1(h1, h2)`: Computes the output layer activation.
- `mse(y_expected, y_predicted)`: Computes the Mean Squared Error between the expected and predicted output.
- `mse_derivative(y_expected, y_predicted)`: Computes the derivative of MSE with respect to the predicted output.

### 7. Output
The output will display:
- The values of the hidden layer activations (`h1` and `h2`).
- The predicted output before and after the weight update.
- The Mean Squared Error before and after the weight update.
- The updated weights.

## Conclusion
This code demonstrates the basic principles of neural networks, including forward propagation, backpropagation, and weight updates using gradient descent. It can be expanded to more complex networks and datasets.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
