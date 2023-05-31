# Image Focus Recognition Neural Network
An image focus recognition A.I. using Neural Networks


THE .h5 FILE THAT CONTAINS THE PRE-TRAINED MODEL IS TOO BIG TO UPLOAD TO GITHUB!

But you can download it via this google drive link: https://drive.google.com/file/d/1F4hOER7-8GwNaRpQjkEx9IFwi4GCWcL5/view?usp=sharing


## 1. Introduction
### 1.1	Motivation
Photographs are used to capture and record detailed still moments of different scenarios, environments or subjects for a wide variety of purposes. For example, taking a high quality picture of a large celestial body, like the moon in order to see and study its craters for scientific research, to even capturing microscopic images of tiny micro-organisms to study how they interact with their surrounding world, to even just capturing a special moment in a person’s life, like a wedding, or a vacation. Regardless of the application of the images, it is important to capture as much detail as possible, but this detail can be lost when the image is out-of-focus; Not only that, it could also render the image unusable. Machine learning has been used in the past not only for binary classification of in- and out-of-focus images, but also the classification of different kinds of blur. Our goal for this project was to classify images that are in-focus and out-of-focus using Neural Networks. This could have useful applications in a camera focusing system, aiding in either manually focusing the lens, or having it connected to a motorised focusing system and adjusting focus automatically.

### 1.2	Related Work
In past work on in- and out-of-focus image classification, there are certain methods that have been used more frequently than others.  One of the most common practices is the use of support vector machines (SVMs) on gradient magnitudes. Support vector machines can be used to classify how blurry images are by first converting input images to grayscale and calculating their gradient magnitudes.  The gradient magnitudes are acquired by calculating the derivatives of the image to determine the rate at which the intensity of the image changes, often by using a Laplacian or Sobel operator (Tomasz Szandala, 2020).  Images with sharper edges such as in-focus images will have larger gradients, as there are harsher changes in intensity at their edges whereas out-of-focus images will have smaller gradients due to their softer and more gradual changes in intensity.  Support vector machines then come into play with these gradient magnitudes by mapping them to a higher dimensional space, then determining a hyperplane in this space that most evenly separates in- and out-of-focus data points.  After training, new data points are categorised based on which side of the hyperplane they land on after being mapped to the higher dimensional space (Hsu and Chen, 2008).

Another highly popular method of detecting blurred images is convolutional neural networks (CNNS).  Convolutional neural networks are capable of being more robust compared to other methods; they are able to work with larger scale images in an efficient manner, and are well suited to object localization such as identifying when there is blur in only a certain region of an otherwise sharp image (Senaras et al., 2018).  In addition, compared to certain other methods, CNNs do not require the user to manually determine a value to act as the threshold between the two classes, which can be quite task dependent. Convolutional neural networks have been used for a variety of different image recognition tasks, such as identifying whether microscopy images are high quality in-focus images (Yang et al., 2018).




## 2. Methods
We plan on using a Convolutional Neural Network (CNN).  CNNs are robust and are well suited to image recognition and classification tasks, as they can detect features regardless of where they are in the image.  This will be especially important in detecting blur that is localised in certain areas of an image. The CNN we made consists of three convolutional layers with ReLU activation, each followed by a max-pooling layer. After flattening the features, a fully connected layer with 128 nodes and ReLU activation is used, along with a 50% dropout to reduce overfitting. Finally, the output layer consists of a single node with a sigmoid activation function.

### 2.1	Dataset
The dataset that we used in our experiments is the Kaggle blur dataset (https://www.kaggle.com/datasets/kwentar/blur-dataset) uploaded by Aleksey Alekseev.  This dataset consists of three types of images, sharp images, defocused images, and motion blurred images, each with 350 images for a total of 1050 images.  The images of the different classes are of the same scenes so as to have the image quality of the different classes be their only discerning feature.


### 2.2	Benefits and drawbacks
Benefits:
CNNs are particularly good for image classification tasks, so using them benefits the overall functionality of the Neural Network.
Having a binary output is particularly helpful for having significantly lower compute times. 
Drawbacks:
It may sometimes be unclear if an image is in focus due to personal preference. These are few and only occur with more complex images that contain multiple subjects in the image. If the photographer captured the image with the intention of having that particular subject in-focus but not the other, then it could be considered “in-focus”

### 2.3	Evaluation/validation strategy
Our qualifications for the network are as follows; The Neural Network should be primarily evaluated based on the accuracy it obtains after it’s been trained and tested. Not only that, but it should also be able to accurately categorise a newly given image. If the Neural Network can do that and keep doing it consistently without error, then we’ll consider it a valid Neural Network that satisfies our goal. Furthermore, we will also be testing several configurations in the Neural Network to find the settings that best optimise the neural network so that it is as accurate as possible while also being computationally efficient. 




## 3. Results
### 3.1	Convolutional Kernel Size
We performed hyperparameter tests on the convolutional kernel size, and averaged then over 5 runs of the environment. The number of epochs was set to 100 and batch size was set to 32.

Convolutional kernel size:
3x3
5x5
7x7

Accuracy on test set:
0.90625
0.859375
0.421875


The accuracy for the 3x3 convolutional kernel size clearly showed more accurate results compared to other kernel sizes, and so we stuck to 3x3 kernels for future testing.

### 3.2	Accuracy Comparison of Focused and Unfocused Test Sets
We split the test set into two separate sets, one for focused images and the other for unfocused images.  When tested on both sets, the test set of unfocused images gave an accuracy of 0.8125, while the test set of focused images gave an accuracy of 1.0.
While the accuracy for the different classes of images is significantly different, this is actually beneficial.  Because we are looking for in-focus images when taking pictures, it is more important that the accuracy for discerning when an image is in-focus be higher compared to discerning when an image is out of focus.  If it believes that an out-of-focus image is in-focus, that image will have to be deleted and the photo retaken, which is not very costly. However, if a perfectly good in-focus image was rejected for being believed to be out-of-focus then that image is lost forever, and is a much more costly loss.

### 3.3	Number of Epochs
We then tested how the number of epochs affected the test set accuracy;  Increasing the number of epochs from 100 to 1000 epochs yielded a noticeable improvement, going from an accuracy of 0.90625 to 0.9874, making our model significantly more accurate. 




## 4. Discussion
### 4.1	Limitations or biases
While testing the model we found a bit of a complication; In some very specific scenarios, the categorization of in-focus/out-of-focus isn’t clear. These occurred only in complex cases with multiple potential subjects and it was due to the fact that it wasn’t immediately obvious what the intended subject was. The Neural Network however still managed to identify when at least 1 subject was in-focus, and going beyond that is an entirely different task and surpasses the complexity of this model at the present time.

### 4.2	Implications of the work
Given that the model manages to be accurate while still being quick means that the model works properly and up to our standards. Our potential goal of having this applied to a camera to be used in an auto-focusing system is possible due to the quick and efficient performance of the model. 


### 4.3	Analysis of results/outcome with respect to objectives
Our main objective was to have a model that could differentiate between images that are in-focus and out-of-focus and we can say that the model can accurately do as such. Not only that, but it’s also able to do it quickly, and as such, I connected it to a Raspberry Pi 4 with a HQ Camera Module and applied it there by feeding the live feed of the sensor to the Neural Network. The entire setup and process with this was a bit of a botch, but nonetheless, it still managed to function in my testing. This confirms one of our initial potential purposes of this model, which is that it can be used in a camera to aid in taking a focused image. 

### 4.4	Potential improvements/future work
The Neural Network could obtain more complexity by being modified to also recognize whether an image contains blur, or other image features like chromatic aberration, lens distortions, etc…  This wouldn’t have the same practical applications as in and out of focus for a focusing system,  but it could have other potential benefits in aiding in capturing a better image.


## References
Hsu, P., Chen, BY. (2008). Blurred Image Detection and Classification. In: Satoh, S., Nack, F., Etoh, M. (eds) Advances in Multimedia Modeling. MMM 2008. Lecture Notes in Computer Science, vol 4903. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-77409-9_26

Senaras, C., Niazi, M. K. K., Lozanski, G., & Gurcan, M. N. (2018). DeepFocus: Detection of out-of-focus regions in whole slide digital images using deep learning. PloS one, 13(10), e0205387. https://doi.org/10.1371/journal.pone.0205387

T. Szandała, "Convolutional Neural Network for Blur Images Detection as an Alternative for Laplacian Method," 2020 IEEE Symposium Series on Computational Intelligence (SSCI), Canberra, ACT, Australia, 2020, pp. 2901-2904, doi: 10.1109/SSCI47803.2020.9308594.

Yang, S.J., Berndl, M., Michael Ando, D. et al. Assessing microscope image focus quality with deep learning. BMC Bioinformatics 19, 77 (2018). https://doi.org/10.1186/s12859-018-2087-4


