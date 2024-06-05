# digits

Project Information
    ```
    - Dataset: MNIST dataset
    - Objective: Classify handwritten digits (0-9) using image data.
    - Skills: Image preprocessing, convolutional neural networks (CNNs), model evaluation. 
    ```

Steps taken to implement

    1. Setup the Environment
    ```
        pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
    ```

    2. Import Libraries and Load the Dataset
    3. Data Preprocessing
    4. Build the Model
    5. Train the Model
    6. Evaluate the Model
    7. Make Predictions and Analyze the Results


Sequential Model

| Layer (type)                  | Output Shape       | Param # |
|-------------------------------|--------------------|---------|
| conv2d (Conv2D)               | (None, 26, 26, 32) | 320     |
| max_pooling2d (MaxPooling2D)  | (None, 13, 13, 32) | 0       |
| flatten (Flatten)             | (None, 5408)       | 0       |
| dense (Dense)                 | (None, 128)        | 692,352 |
| dense_1 (Dense)               | (None, 10)         | 1,290   |

**Total params:** 693,962 (2.65 MB)

**Trainable params:** 693,962 (2.65 MB)

**Non-trainable params:** 0 (0.00 B)



**Per Epoch Training Loss and Accuracy**
```
Epoch 1/10
300/300 - 4s - 14ms/step - accuracy: 0.9311 - loss: 0.2492 - val_accuracy: 0.9735 - val_loss: 0.0860
Epoch 2/10
300/300 - 4s - 13ms/step - accuracy: 0.9780 - loss: 0.0734 - val_accuracy: 0.9799 - val_loss: 0.0612
Epoch 3/10
300/300 - 4s - 14ms/step - accuracy: 0.9857 - loss: 0.0487 - val_accuracy: 0.9838 - val_loss: 0.0464
Epoch 4/10
300/300 - 4s - 14ms/step - accuracy: 0.9888 - loss: 0.0376 - val_accuracy: 0.9858 - val_loss: 0.0462
Epoch 5/10
300/300 - 4s - 14ms/step - accuracy: 0.9915 - loss: 0.0279 - val_accuracy: 0.9837 - val_loss: 0.0488
Epoch 6/10
300/300 - 4s - 14ms/step - accuracy: 0.9934 - loss: 0.0218 - val_accuracy: 0.9867 - val_loss: 0.0439
Epoch 7/10
300/300 - 4s - 14ms/step - accuracy: 0.9955 - loss: 0.0163 - val_accuracy: 0.9870 - val_loss: 0.0384
Epoch 8/10
300/300 - 4s - 14ms/step - accuracy: 0.9961 - loss: 0.0133 - val_accuracy: 0.9855 - val_loss: 0.0448
Epoch 9/10
300/300 - 4s - 14ms/step - accuracy: 0.9972 - loss: 0.0102 - val_accuracy: 0.9866 - val_loss: 0.0417
Epoch 10/10
300/300 - 4s - 14ms/step - accuracy: 0.9979 - loss: 0.0077 - val_accuracy: 0.9826 - val_loss: 0.0587
Test loss: 0.058656297624111176
Test accuracy: 0.9825999736785889
```


