# MNIST Handwritten Digit Recognition

This project aims to classify **MNIST handwritten digits** using a simple **Artificial Neural Network (ANN)**.

## 🚀 Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- Matplotlib

## 📌 Project Description
This project uses the **MNIST dataset** to classify handwritten digits from 0 to 9 using an **Artificial Neural Network (ANN)** model.

The model is trained using the **TensorFlow/Keras** library and consists of 3 layers in total, including 2 hidden layers:

1. **Input Layer:** 784 neurons (flattened 28x28 pixel images)
2. **Hidden Layers:**
   - 128 neurons, **ReLU activation function**
   - 64 neurons, **ReLU activation function**
3. **Output Layer:** 10 neurons (**Softmax activation function**) - Each neuron represents a digit class.

---

## 📂 File Structure

- `project.py` → Main code file
- `README.md` → Project documentation

---

## 🔧 Installation and Execution

### 1️⃣ Install Dependencies
Before running the project, install the required libraries:

```bash
pip install tensorflow numpy matplotlib
```

### 2️⃣ Run the Project
You can execute the Python file using the following command in the terminal or command prompt:

```bash
python project.py
```

---

## 📊 Model Training

The model is trained for **10 epochs** with a **batch size of 32**. During training, **80% of the data is used for training and 20% for validation**.

```python
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
```

After training, the accuracy on the test set is calculated:

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")
```

---

## 📷 Prediction and Visualization
To see the model’s predictions alongside actual values, use the following code:

```python
index = 8675
plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
plt.title(f"Prediction: {predicted_labels[index]}, Actual: {y_test[index]}")
plt.show()
```
This code displays an image of the digit at the specified **index**, along with its actual and predicted values.

---

## 📌 Conclusion
This project contains a basic **Artificial Neural Network (ANN) model** that recognizes MNIST handwritten digits. The model’s accuracy can be improved by using more advanced architectures such as **Convolutional Neural Networks (CNNs)**.

📩 **Contributions:** If you want to contribute to the project, feel free to open a pull request or share your suggestions! 🎯

