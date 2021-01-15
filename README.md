# ASL-Detection-Using-A-Neural-Network
ASL Fingerspelling Translation to Text Using a CNN

# Project Description
#### This is a group project for my machine learning class. This program accepts images of ASL fingerspelling and classifies the input into one of 29 symbols: the 26 letters of the alphabet, space, delete, and nothing. We hoped to create something that would aid people who are deaf or hard of hearing and expand the accessibility of online products that target those groups. Towards this goal we combined two data sets we obtained from kaggle for a total of 164,500 images which we processed and used in our model training, validation, and testing.

#### Data sets: 
#### https://www.kaggle.com/grassknoted/asl-alphabet
#### https://www.kaggle.com/kuzivakwashe/significant-asl-sign-language-alphabet-dataset


# Data Preprocessing
#### Since much of our data was quite similar, we wanted to introduce some additional noise into our images to help prevent our model from overfitting. To do so, we adjusted the images via horizontal shifting, vertical shifting, and zooming to a random extent of 0 to 20%. 

#### We then wanted to segment each image into the object of focus and other elements which was especially important given the limited diversity in our images. We did not want to simply transform each image into black and white based on the RGB values of the pixels as variations in lighting and the background could erase portions of the hands. Instead we opted to use the YCrCb color space for segmenting our images as this space measures in order luminance, blue chroma (blueness controlled for luminance), and red chroma (redness controlled for luminance). The YCrCb space explicitly accounts for brightness and can recognize differences between pixels of the same brightness or find similarities between pixels of differing brightnesses that might be missed by simply using the RGB space and thus might prove advantageous for hand segmentation.

#### We then used a k-means approach to sort the pixels of each image into one of  two categories: hand or background. The k-means algorithm does not transform the image into blsack and white but rather into one of these two categories, but this process can be visualized by black and white pixels as shown below.


# Model Validation
#### After preprocessing our data, we separated our data randomly into three smaller subsets. We designated 60% of our data as a training data set, 20% as a validation data set, and 20% as a testing data set. Each model architecture was subsequently trained on our training data set and tested on the validation data set. Based on each model’s performance on the validation data set we would experiment with different hyperparameters and different combinations of layers with the goal of improving this performance. After we had selected our model, we used the test data set as an estimate of how the model would generalize and perform with data it had not previously encountered. 


# Model Architecture
#### Although there currently exists no set formula for building the ideal CNN for any given data set, there are general guidelines to abide by that we could infer by examining the architectures of various state of the art CNN models such as CifarNet, AlexNet, and GoogLeNet. We decided to alternate between convolutional and pooling layers, followed by some dense fully connected layers, and then we would use a dense layer with 29 units as the output layer. Due to computing and time constraints, we decided to experiment with mainly the amount of alternating convolutional and pooling layers and the amount of dense layers, and the filters and units hyperparameters for the convolutional and dense layers respectively. All of the convolutional and dense layers in our model used the reLu activation function, with the exception of our output layer which instead used the softmax activation function.

#### We first began by experimenting with both the number of filters per convolutional layer and the amount of alternating convolutional and pooling layers in the model. The four tables below show the model’s performance across ten epochs for the filter variations we tested. As shown in the data, generally performance for each set of alternating sets of convolutional and pooling layers improved as we increased the filter sizes across the layers. Performance also generally increased with additional convolution and pooling layers. Increasing the amount of filters greatly increases the time it takes for the model to train though, as the 64, 128, 128 variant of table 3 took about 2.5 hours just to fit while the 16, 32, 64 variant took only 15 minutes to fit. Therefore, when taking into consideration both performance and the time it took for the model to train, we opted to 3 sets of alternating convolution and pooling layers with 16, 32, and 64 filters for the 3 convolutional layers respectively.

#### As shown in the data, the addition of a dense layer greatly increased the model’s performance but the relationship between additional units in a layer and overall performance is much less transparent. Adding a second dense layer did not seem to greatly affect the model’s performance, so we opted to just use a single dense layer with 1000 units since the 1000 unit variant did perform slightly better than its 500 and 250 unit counterparts.


# Regularization
#### We experimented with Batch Normalization, Dropout, and L2 regularization. Many of the methods we tested did in fact result in increased generalization performance, but others actually decreased generalization performance. Additionally, some methods worked better on their own while others worked better when combined with other techniques. By using both L2 regularization in tandem with dropout layers, we achieved a final generalization rate of 97.43% for our model.

# Results
#### Our model had an accuracy of 97.43% on our test data set. The most common mistake was the model misinterpreting the asl sign for “r” as “u”, which is understandable as the asl signs for these two letters are quite similar.

