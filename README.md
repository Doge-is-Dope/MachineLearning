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

```model.fit``` trains the model by providing the features (inputs) and labels (outputs) by using **gradient descent**.

**Loss function** is the function that measures how good or bad the model is during each iteration.

During the training process, the goal is to minimize the loss function.

