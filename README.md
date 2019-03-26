# Machine Learning

# Celsius to Fahrenheit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chunchiehliang/MachineLearning/blob/master/Celsius_to_Fahrenheit.ipynb)


```python
l0 = tf.keras.layers.Dense(units=1, input_shape=[1]) 
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
model.predict([100.0])
```

### Keywords
- Feature: The input(s) to our model
- Examples: An input/output pair used for training
- Labels: The output of the model
- Layer: A collection of nodes connected together within a neural network.
- Model: The representation of your neural network
- Dense and Fully Connected (FC): Each node in one layer is connected to each node in the previous layer.
- Weights and biases: The internal variables of model
- Loss: The discrepancy between the desired output and the actual output
- MSE: Mean squared error, a type of loss function that counts a small number of large discrepancies as worse than a large number of small ones.
- Gradient Descent: An algorithm that changes the internal variables a bit at a time to gradually reduce the loss function.
- Optimizer: A specific implementation of the gradient descent algorithm. (There are many algorithms for this. In this course we will only use the “Adam” Optimizer, which stands for ADAptive with Momentum. It is considered the best-practice optimizer.)
- Learning rate: The “step size” for loss improvement during gradient descent.
- Batch: The set of examples used during training of the neural network
- Epoch: A full pass over the entire training dataset
- Forward pass: The computation of output values from input
- Backward pass (backpropagation): The calculation of internal variable adjustments according to the optimizer algorithm, starting from the output layer and working back through each layer to the input.

### Training Process
1. Forward pass: In this stage, the input data (**feature**) is fed to neural network. Then the model predicts a value.
2. Loss calculation: The model calculates the difference (**loss**) between the predicted value and the correct value (**label**). The value of the loss is calculated using a loss function, which is specified with the loss parameter in ```model.compile()```.
3. Optimization: After the loss value is calculated, the internal variables (**weights** and **biases**) of all the layers of the neural network are adjusted to minimize the loss to make the prediction closer to the correct value. The process is called **gradient descent** which specifies the algorithm by the parameter "optimizer" like ```Adam```.


**Loss function** is the function that measures how good or bad the model is during each iteration. During the training process, the goal is to minimize the loss function.

```model.fit``` trains the model by providing the features (inputs) and labels (outputs) by using **gradient descent**.






### Reference
Google Tensorflow 2.0 on Udacity, 2019

