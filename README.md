# **Behavioral Cloning Project**
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: report_images/LeNet.png
[image3]: report_images/center.JPG
[image4]: report_images/left.JPG
[image5]: report_images/right.JPG
[image6]: report_images/hist1.JPG
[image7]: report_images/hist2.JPG
[image8]: report_images/history.JPG

---
## Model Architecture and Training Strategy

### Model Architecture

My model consists is based off of the LeNet Architecture but fully retrained for my training data with a different optimizer and one output. The code for generating and training models are all in `model.py`.

As shown below, the LeNet architecture was used with no major modifications, but pre-processing and modifications were added.
![][image1]

I also tried the Nvidia self-driving car architecture using deep learning interchangably and compared the two's performance.
<img src="report_images/Nvidia.png" width="300">

Both tested models include ReLU layers to introduce nonlinearity [Code line 105 (and all other convolutional layers)], and the data is normalized in the model using a Keras lambda layer [Code line 104 and 135]. 

### Reduce Overfitting
Both models contain dropout layers in order to reduce overfitting [Code line 111 (and after all other fully connected layers)]. 

Max Pooling was also used in the LeNet architecture [Code lines 106 and 108], reducing the size of hte feature map between after each convolutional layer. 

After validating the training on different data sets (or a mix of them) to ensure the model isn't overfitting, the model was used to autonomously drive the car successfully through the track without leaving the drivable area.

### Model Parameter Tuning
Using an Adam optimizer [Code lines 119 and 154], the learning rate was not tuned manually.

### Training Data
Training data was strategically created:
* Center lane driving for 4 laps
* Reverse direction driving for 2 laps
* One lap of only recording smooth corner turns
* One lap of only recording recovery from different edges

Further details are below in data acquisition. 

## Training Strategy

### 1. Solution Design Approach
The overall strategy for deriving a model architecture was to start simple and then try different additions/pre-processing techniques before implementing a full architecture from elsewhere (LeNet and Nvidia in this case).

My first step was to use only a single fully connected layer. It was evidently inaccurate as it only outputted +_25 deg, but it did move the car in the simulator.

After confirming the basic model implementation, both the LeNet and Nvidia architecutres were implemented, and then the data was split into a randomized training and validation set for steering angle data.

By looking at the mean-squared loss error visualized over the epochs, I can see if I was overfitting (training error << validation error) or underfitting (large error). This led me to either add dropout layers or improve the complexity of the model (add more convolutions or fully connected layers). An example of the loss visualization is shown below:

![][image8]

Additionally, I found that data was extremely important, as poor data will not only produce bad driving, but will lead the model to go off the track (e.g. go straight through a turn).

To combat the overfitting, I modified the model to not overfit, provided increasing amounts of data, and generated more data through flipping and using more cameras.

The histogram of angles in my training data is shown below:

![][image6]

By flipping the data alone, the data is symmetrical about 0 degrees as shown below:

![][image7]

After numerous tests on the racetrack, the LeNet architecture performed more consistently on recovery, proving to be more consistent than the model built off of Nvidia. Often, the Nvidia model will go straight when it should turn despite being trained on the same data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture
The final model architecture [Code lines 95 to 124] consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Cropping        		| 65x320x3 RGB image   							| 
| Normalization      | 65x320x3 RGB image   							| 
| Convolution 5x5  	| 1x1 stride, valid padding, outputs 61x316x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 31x31x6 					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 27x27x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 					|
| Flattening	        | Array of 1176 (= 14x14x6) 						|
| Fully connected		| Output of size 120							|
| RELU					|												|
| Dropout				|		50% drop rate										|
| Fully connected		| Output of size 86								|
| RELU					|												|
| Dropout				|		50% drop rate										|
| Fully connected		| Output of size 1								|

### 3. Creation of the Training Set & Training Process
As described before, the reasoning behind the strategic training data collection was:
* Center lane driving - ensure the car stays near the center of the road
* Reverse direction driving - improve generalization of the model
* Only smooth corner turns - improve the ratio of higher angle turning to lower the dominance of small angles as most driving is straight
* Only recovery from sides - add more data to learn recovering from sides during straight roads and turns of different line colors

Afterwards, the data from all three cameras on the simulated car was used and they were all flipped to double the data set and prevent left-turn biasing due to the counter-clockwise driving.

An example of the data is shown below:

Center camera:

![][image3]

Left Camera: 

![][image4]

Right Camera:

![][image5]

After the collection process, I had approximately 48000 (= 6 * 8000) data points due to the generated data. I then preprocessed this data by cropping 70 pixels off the top and 25 pixels off the bottom to remove the background and the front of the car to prevent the model fitting to the background environment (trees, water, etc.) and also to improve its generalizability and robustness.

Finally, the data is randomized and 20% is used as a validation set during training, and also to be used as loss visualization, which is used to determine over/underfitting
