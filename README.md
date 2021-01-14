# ASL-Detection-Using-A-Neural-Network
ASL Fingerspelling Translation to Text Using a CNN

# Project Description
#### This is a group project for my machine learning class. This program accepts images of ASL fingerspelling and classifies the input into one of 29 symbols: the 26 letters of the alphabet, space, delete, and nothing. We hoped to create something that would aid people who are deaf or hard of hearing and expand the accessibility of online products that target those groups. Towards this goal we combined two data sets we obtained from kaggle for a total of 164,500 images which we processed and used in our model training, validation, and testing. 

# Data Preprocessing
#### Since much of our data was quite similar, we wanted to introduce some additional noise into our images to help prevent our model from overfitting. To do so, we adjusted the images via horizontal shifting, vertical shifting, and zooming to a random extent of 0 to 20%. 

#### We then wanted to segment each image into the object of focus and other elements which was especially important given the limited diversity in our images. We did not want to simply transform each image into black and white based on the RGB values of the pixels as variations in lighting and the background could erase portions of the hands. Instead we opted to use the YCrCb color space for segmenting our images as this space measures in order luminance, blue chroma (blueness controlled for luminance), and red chroma (redness controlled for luminance). The YCrCb space explicitly accounts for brightness and can recognize differences between pixels of the same brightness or find similarities between pixels of differing brightnesses that might be missed by simply using the RGB space and thus might prove advantageous for hand segmentation.

#### We then used a k-means approach to sort the pixels of each image into one of  two categories: hand or background. The k-means algorithm does not transform the image into blsack and white but rather into one of these two categories, but this process can be visualized by black and white pixels as shown below.


# Model Architecture and Validation

# Results
