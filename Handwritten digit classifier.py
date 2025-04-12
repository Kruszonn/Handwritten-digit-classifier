import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import cv2
import os

dataset = fetch_openml('mnist_784', version=1)

X = dataset.data.to_numpy() / 255.0
Y = dataset.target.to_numpy().astype(int)

encoder = OneHotEncoder(sparse_output=False)
Y_oh = encoder.fit_transform(Y.reshape(-1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_oh, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}, {Y_train.shape}")
print(f"Test set shape: {X_test.shape}, {Y_test.shape}")

def inicjalizacja_parametrów(layer_dims):
    np.random.seed(1)
    parameters = {}
    warstwy = len(layer_dims)

    for warstwa in range(1, warstwy):
        parameters[f"W{warstwa}"] = np.random.randn(layer_dims[warstwa], layer_dims[warstwa - 1]) * np.sqrt(2 / layer_dims[warstwa - 1])
        parameters[f"b{warstwa}"] = np.zeros((layer_dims[warstwa], 1))

        assert parameters[f"W{warstwa}"].shape == (layer_dims[warstwa], layer_dims[warstwa - 1])
        assert parameters[f"b{warstwa}"].shape == (layer_dims[warstwa], 1)

    return parameters

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2 


    for l in range(1, L):
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]   
        Z = np.dot(W, A) + b
        A = relu(Z)
        caches.append((A, Z)) 

    WL = parameters[f"W{L}"]
    bL = parameters[f"b{L}"]
    ZL = np.dot(WL, A) + bL
    AL = softmax(ZL)
    caches.append((AL, ZL))
        
    return AL, caches   

def compute_loss(AL, Y):
    m = Y.shape[1]
    loss = -(1/m) * np.sum(Y * np.log(AL + 1e-8))
    loss = np.squeeze(loss)
    return loss

def relu_derivative(Z):
    return (Z > 0).astype(float)

def backward_propagation(X, Y, AL, caches, parameters):
    grads = {}
    L = len(parameters) // 2 
    m = Y.shape[1]
    
    dZL = AL - Y
    
    A_prev = caches[L-2][0]  
    grads[f"dW{L}"] = (1/m) * np.dot(dZL, A_prev.T)
    grads[f"db{L}"] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
    
    dA = np.dot(parameters[f"W{L}"].T, dZL)
    
    for l in reversed(range(1, L)):
        Z_curr = caches[l-1][1]
        dZ = dA * relu_derivative(Z_curr)
        
        if l > 1:
            A_prev = caches[l-2][0]
        else:
            A_prev = X
            
        grads[f"dW{l}"] = (1/m) * np.dot(dZ, A_prev.T)
        grads[f"db{l}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        if l > 1:
            dA = np.dot(parameters[f"W{l}"].T, dZ)

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 

    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    return parameters

def train_model(X, Y, layer_dims, learning_rate=0.01, num_epochs=1000, print_loss=False):
    np.random.seed(1)
    
    parameters = inicjalizacja_parametrów(layer_dims)
    costs = []

    for i in range(num_epochs):
        AL, caches = forward_propagation(X, parameters)
        
        loss = compute_loss(AL, Y)
        
    
        grads = backward_propagation(X, Y, AL, caches, parameters)
        
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_loss and i % 100 == 0:
            print(f"Cost after iteration {i}: {loss:.4f}")
            costs.append(loss)
            
    return parameters, costs

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    
    predictions = np.argmax(AL, axis=0)
    
    return predictions

def save_model(parameters, filename="model_parameters.npz"):
    np.savez(filename, **parameters)
    print(f"Model saved to {filename}")

def load_model(filename="model_parameters.npz"):
    data = np.load(filename, allow_pickle=True)
    parameters = {}
    for key in data.files:
        parameters[key] = data[key]
    print(f"Model loaded from {filename}")
    return parameters

def preprocess_image(image_path):
   
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    avg_intensity = np.mean(gray)
    if avg_intensity > 128:  
        gray = 255 - gray  
    
    
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    normalized = resized / 255.0
    
    flattened = normalized.reshape(1, 784)
    
    return flattened

def test_mnist_image(X_test, Y_test, parameters, index=0):
    # Get a single test image
    test_image = X_test[index:index+1]
    true_label = np.argmax(Y_test[index:index+1])
    

    prediction = predict(test_image.T, parameters)
    
    # Visualize
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image.reshape(28, 28), cmap='gray')
    plt.title(f"MNIST Test Image\nTrue Label: {true_label}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(test_image.reshape(28, 28), cmap='gray')
    plt.title(f"Prediction: {prediction[0]}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return prediction[0], true_label


def predict_custom_image(image_path, parameters):
    """Predict digit in a custom image"""
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Predict
    prediction = predict(processed_image.T, parameters)
    
   
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #RGB to matplotlib
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    processed_img = processed_image.reshape(28, 28)
    plt.imshow(processed_img, cmap='gray')
    plt.title(f"Processed Image\nPrediction: {prediction[0]}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return prediction[0]

layer_dims = [784, 128, 64, 10]

train_new_model = True 

if train_new_model:
    print("Training new model...")
    parameters, costs = train_model(X_train.T, Y_train.T, layer_dims, 
                                   learning_rate=0.01, num_epochs=1000, print_loss=True)
    
    
    predictions = predict(X_test.T, parameters)
    true_labels = np.argmax(Y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(costs) * 100, 100), costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Learning Curve (Cost vs. Iterations)')
    plt.grid(True)
    plt.show()
    
    
    save_model(parameters)
else:
    
    parameters = load_model()

#predict image 
#predict_custom_image('image1.jpg', parameters)



correct = 0
for i in range(10):
    random = np.random.randint(0, X_test.shape[0])
    pred, true = test_mnist_image(X_test, Y_test, parameters, index=random)
    print(f"Image {i+1}: Prediction: {pred}, True Label: {true}")
    if pred == true:
        correct += 1
print(f"Accuracy on sample: {correct/10:.2f}")



