# **Traffic Sign Recognition** 

## Main Files
* Souce Code: Traffic_Sign_Classifier.ipynb
* Source Code Export Format: Traffic_Sign_Classifier.html
* Project Instruction: ProjectDescrition.md
* output folder: all the result images, models
* test_imgs folder: all the test images of german traffic signs found on web
* PS: training dataset is in ../data folder. Not available here


## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/signs.jpg "Visualization Signs"
[image2]: ./output/histogram.jpg "Visualization Histogram"
[image3]: ./output/convert_image.jpg "Image Enhancement"
[image4]: ./output/augment.jpg "Augment"
[image5]: ./output/histogram_include_augment.jpg "Histogram with Augment"   
[image6]: ./output/test_imgs.jpg "Test Images"   

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used pickle library to load data (in CODE CELL 1 of Traffic_Sign_Classifier.ipynb).

I used python library to calculate summary statistics of the traffic
signs data set (in CODE CELL 2):

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

 In CODE CELL 4, we can see the count for each class, where max count is 2010, min count is 180, and mean count is 809.

### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. One image for each class is displayed.
![alt text][image1]

This is a bar chart showing how many training data in each class.
![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to convert the color images to grayscale 
because [LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) suggests one channle i.e. grayscale image outperforms color images. 
In paper, it converted color image to YUV color space, and use the normalized Y channel.

I noticed in opencv library, the equation for converting RGB to Gray is exactly the same as calculating Y in YUV, i.e.
[[ref1]](https://answers.opencv.org/question/57947/understanding-yuv-to-bgr-color-spaces-conversion/)
[[ref2]](https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html)
```
Gray or Y = 0.299R + 0.587G + 0.114B
```       
Then I equalized histogram of the gray image for image enhancement (in CODE CELL 6).                                         
[[ref3]](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html).  

Here is an example of a traffic sign image before and after grayscaling and histogram equlization.
![alt text][image3]

I decided to generate additional data because 
* some class (for example class 0) only has around 200 samples, 
while other class (e.g. class 1) has around 2000 samples. This makes training data not well balanced.
* Different brighness, contrastness, rotation angles, scalling etc. of training data are expected to improve the model performance.

To add more data to the the data set, I used the following techniques (in CODE CELL 9):
if the samples count of the given class is below average, add 4 new images based on this image, 
including clockwise and anti clockwise 15 degrees rotation, and vertically and horizontally scaled images

This bumped the total training data from 34799 to 71275. The new histogram is as below, which seems better balanced among classes.
![alt text][image5]  

Here is an example of an original image and its augmented images:
![alt text][image4]   

In the model training section, the training dataset with augmented data outperforms. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		 | Layer Details         |     Description	        					| 
|:----------------------:|:---------------------:|:--------------------------------------------:| 
|      1                 | Input                 | 32x32x1 gray image   						| 
|                        | Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6 	|
|                        | RELU			         |												|
|                        | Max pooling	         | 2x2 stride, outputs 14x14x6  				|
|      2                 | Convolution 5x5	     | 1x1 stride, valid padding, outputs 10x10x16  |
|                        | RELU			         |												|
|                        | Max pooling	         | 2x2 stride, outputs 5x5x16  			    	|
|                        | Flatten   	         | Output = 400  				                |
|      3                 | Fully connected		 | Output = 120        							|
|                        | RELU			         |												|
|                        | Dropout			     | Dropout=0.3									|
|      4                 | Fully connected		 | Output = 84        							|
|                        | RELU			         |												| 
|                        | Dropout			     | Dropout=0.3									|    
|      5                 | Fully connected		 | Output = 43      							|    
|                        | Softmax				 |           									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used an Adamd Optimizer, batch size 128, epochs 50, learing rate 0.001. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

In CODE CELL 16, I used training data with augmented images, since it works better than the training dataset without augmented data.
If you'd like to try training model without augmented data, don't use CODE CELL 16, and use CODE CELL 17 instead.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.959
* test set accuracy of 0.929

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture was the model in the LeNet Lab in this course, since it is known for good performance.

* What were some problems with the initial architecture?

Not really. Everything works fine. I did a little research out of curiosity. 
[This project](https://github.com/lijunsong/udacity-traffic-sign-classifier) increases the filter depth in order to catch more structures/features. 
This idea is very interesting to me. I tried the setting mentioned in this project, but I didn't see significant improvement in my work.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Added dropout to prevent overfitting.

I used dropout = 0.3 to prevent overfitting.

Also plotted a learning curve for training accuracy and validation accuracy. It converges around 30 epochs. This curve also helps to see whether it is under or over fitting.

* Which parameters were tuned? How were they adjusted and why?

Based on the Accuracy curve, the epoch seems stable after 30+. Decided to set it to 50.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

 Used dropout in layer 3 and layer 4 to prevent overfitting.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
Mainly used the LeNet lab provided by the course. Didn't make significant change. Pretty happy with 1) the pre processing work for converting to gray image and equalization, 
and 2) add the augmented images which is very practical and useful in real world project.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 16 German traffic signs that I found on the web:

![alt text][image6] 

The 5th image might be difficult to classify because of noisy background and smaller scale.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction (in CODE CELL 23):

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)   						| 
| Speed limit (50km/h)  | Dangerous curve to the right			     	|
| Speed limit (60km/h)	| Speed limit (60km/h)							|
| Bumpy road    		| Bumpy Road					 				|
| General caution		| Bicycles crossing      						|
| ...           		| ...      						|

The model was able to correctly guess 3 of the 4 traffic signs, which gives an accuracy of 75%. This compares favorably to the accuracy on the test set of 0.929.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23th cell of the Ipython notebook.

For the first image, the model is very sure that this is a Speed limit (30km/h) (probability of 0.998). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.998         		| Speed limit (30km/h)  						| 
| 0.001     			| Speed limit (20km/h)							|
| 0.000					| Roundabout mandatory							|
| 0.000	      			| Speed limit (70km/h)					 		|
| 0.000				    | End of speed limit (80km/h)     				|



