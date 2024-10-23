# Neural Network Progress


# 1. Initialize network with random weights and biases

# 2. For EACH training example
for example in training_data:
    # 2a. Compute activations for the entire network

    # 2b. Compute δ for neurons in the output layer using:
    delta_i_L = 2 * (a_hat - a) * relu_prime(z)

    # 2c. Compute δ for all neurons in previous layers:
    delta_i_L = sum(delta_i_L_plus_1 * w_j_i_L_plus_1 * relu_prime(z) for j in range(n))

    # 2d. Compute gradient cost in each weight and bias for training data using δ
    grad_w_j_i_L = delta_j_L * a_i_L_minus_1
    grad_b_i_L = delta_j_L

# 3. Average the gradient w.r.t each weight and bias over the entire training set
avg_grad_w_j_i_L = (1 / n) * sum(grad_w_j_i_L for i in range(n))
avg_grad_b_i_L = (1 / n) * sum(grad_b_i_L for i in range(n))

# 4. Update the weights and biases using gradient descent
w_j_i_L -= eta * avg_grad_w_j_i_L
b_i_L -= eta * avg_grad_b_i_L

# 5. Repeat steps 2-4 till cost reduces to an acceptable level
while cost > acceptable_level:
    # repeat the process
