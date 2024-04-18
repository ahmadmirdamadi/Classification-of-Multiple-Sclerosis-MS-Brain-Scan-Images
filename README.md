# Classification-of-Multiple-Sclerosis-MS-Brain-Scan-Images

This code is for training a convolutional neural network (CNN) model to classify images of multiple sclerosis (MS) brain scans. Here's a breakdown of the steps:

**1. Setup and Data Loading:**
   - Libraries like `numpy`, `pandas`, `os`, etc. are imported for data manipulation and visualization.
   - The code checks if required libraries (`tensorflow` and `keras`) are installed and installs them if not.
   - It downloads the "multiple-sclerosis" dataset from Kaggle and extracts it.
   - The script finds the image files within the dataset and categorizes them based on their folder names (Control/MS in Axial/Sagittal views).

**2. Data Preprocessing:**
   - Image sizes are checked for consistency. 
   - A dictionary is used to map folder names to numerical labels for the machine learning model. 
   - A function is defined to retrieve the corresponding folder name (class label) based on the numerical label.
   - Images are loaded, resized to a standard size (100x100 pixels in this case), and converted into a format suitable for the neural network.
   - The data is split into training, validation, and testing sets.
   - Training data is normalized by dividing each pixel value by 255 (assuming the images use 8-bit grayscale values).

**3. Model Building:**
   - A pre-trained VGG16 model is loaded with weights trained on the ImageNet dataset. 
   - The top layers of VGG16 are frozen (their weights are not updated during training) as they are specialized for general image recognition tasks.
   - A new model is created by adding a GlobalAveragePooling2D layer to summarize the features extracted by VGG16, followed by dense layers with ReLU activation and dropout for regularization.
   - The final layer has 4 neurons (one for each class) and uses softmax activation to predict class probabilities.

**4. Model Training and Evaluation:**
   - The model is compiled with the Adam optimizer, categorical crossentropy loss function (suitable for multi-class classification), and accuracy metrics.
   - Categorical encoding is applied to convert class labels into one-hot vectors for training.
   - The model is trained on the training data with validation data monitored to prevent overfitting.
   - The model's performance is evaluated on the unseen test data, reporting test accuracy.
   - Training and validation loss/accuracy curves are plotted for visualization.

**5. Performance Analysis:**
   - Classification report, precision, recall, and F1-score are calculated to evaluate the model's performance on each class.
   - A confusion matrix is generated to visualize how often the model predicted each class correctly or incorrectly.

**6. Saving the Model:**
   - The trained model is saved for future use.

Overall, this code demonstrates how to transfer learning with a pre-trained CNN model (VGG16) for classifying medical images related to multiple sclerosis.
