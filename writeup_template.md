**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/Nvidia-model.png "Nvidia Model"
[image3]: ./examples/track1-center.png "track1 center"
[image4]: ./examples/track1-left.png "track1 left"
[image5]: ./examples/track1-right.png "track1 right"
[image6]: ./examples/track2-center.png "track2 center"
[image7]: ./examples/track2-left.png "track2 left"
[image8]: ./examples/track2-right.png "track2 right"
[image9]: ./examples/original-data.png "data"
[image10]: ./examples/flipped.png "flipped data"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
Files Submitted & Code Quality

1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

1. Solution Design Approach

   The overall strategy for deriving a model architecture was to predict the steering angle of the simulator.
   
   My first step was to use a convolution neural network model similar to the NVIDIA model I thought this model might be appropriate because it has been used to successfully predit steering angles.
   
   This model consists of a convolution neural network with 5 convolution layers dropouts between the fully connected layers.The model includes RELU layers to introduce nonlinearity (code line 63-74), and the data is normalized in the model using a Keras lambda layer (code line 61). 
   
   In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
   
   To combat the overfitting, I modified the model by adding dropout layers to reduce overfitting.
   
   The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.To improve the driving behavior in these cases, I gathered more data at those spots. 
   
   At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

2. Attempts to reduce overfitting in the model
   
   The model contains dropout layers in order to reduce overfitting (model.py lines 68-73). The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

3. Final Model Architecture

   The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. The final model architecture (model.py lines 60-75) consisted of a convolution neural network with the convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.

   Here is a visualization of the architecture.
   
   ![alt text][image2]

   ![alt text][image1]

4. Creation of the Training Set & Training Process

   The model was trained on an AWS g2.2xlarge EC2 instanceTo capture good driving behavior, I first recorded two laps on track    one using center lane driving. 
   
   Here is an example images of track one.
   
   left
   
   ![alt text][image4]
   
   center
   
   ![alt text][image3] 
   
   right
   
   ![alt text][image5]

   Then I repeated this process on track two in order to get more data points.
   
   Below the are the images of track two.
   
   left
   
   ![alt text][image7] 
    
    center
    
    ![alt text][image6] 
    
    right
    
    ![alt text][image8]

   To augment the data set, I also flipped images and angles thinking that this would generalize my model.
   
   ![alt text][image9]
   
   ![alt text][image10]

   I finally randomly shuffled the data set and put 20% of the data into a validation set. 

   I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by having more epochs did not have any effect on the loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
