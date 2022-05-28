# Computer Vision for fire detection using Convolutional Neural Networks

In this README, you can find an in-depth explanation on the resources needed, as well as an explanation for each section of the code. 
Each section is well commented in order to create a nice environment for the reader to understand everything correctly.

# Code Sections

- Mean and Standard Deviation: First, a basic transformation is applied to the images, which consists of modifying the size of 
all the images to standardize them. DataLoader is then used to separate the images into batches. Then, we calculate the mean 
and standard deviation corresponding to the set of images. These are presented in the form of a list with three values because 
the images are in RGB, thus being a value for each channel.
- Image Pre-processing: Images are again transformed by resizing them to a general 300x300 size, flipping them horizontally on 
a random 50% chance, and finally, normalizing them using the mean and standard deviation values. These images are then separated 
into batches to split them up into three subsets. The first split has a 70 - 30 proportion. Then, the 70% subset is split into 
two new ones consisting of a 75 - 25 proportion. This way, the three subsets represent 52.5%, 17.5%, and 30% respectively.
- Show random images: The first batch of images from the training dataset is then shown on a single plot.
- Network model creation: The neural network's architecture was defined. We used two convolutional layers, two pooling layers and 
three linear layers. The activation function for all layers is the Relu function, except for the last layer who uses a Sigmoid 
function. The loss and optimizer functions used are the Binary Cross Entropy and Stochastic Gradient Descent function, 
respectively.
- Training the CNN: For 15 epochs, we train the model using the training subset. The loss is calculated using Binary Cross-Entropy 
and the optimizer used for this process was SGD (Stochastic Gradient Descent). Loss for each epoch is printed to look at how the 
model is doing after each run is finished.
- Partial Validation: After training the model, data from the first testing subset (which consists of 12.5% of the data) is used 
to validate it. The results obtained from the model are then compared with the true labels to check for the accuracy of the model.
- Evaluation Metrics: The following metrics were used in order to check the model's behavior: Confusion Matrix, recall, precision, 
f1-score, ROC, curve, and ROC-AUC score.
- Final model's validation: Data from the final testing subset (which consists of 30% of the data) is used to validate it. 
The results obtained from the model are then compared with the true labels to check for the accuracy of the model.

# DockerFile:
The dockerfile installs all the libraries that were neccesary for the development of the project.

# DATA:
The data was retrieved from https://www.kaggle.com/datasets/phylake1337/fire-dataset?resource=download
