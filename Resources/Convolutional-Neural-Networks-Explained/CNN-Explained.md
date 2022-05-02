# Convolutional Neural Networks Explained
Computer vision is a subset of machine learning which involves deep learning models learning from digital imagery and videos to complete a given task. The task could be to simply identify the correct animal in a certain picture (classification) or to convert an image to a higher resolution (regression). Computer vision has a multitude of applications and researches are creating new architectures all the time to out perform the last, the simplist however is the Convolutional Neural Network.

A Convolutional Neural Network (ConvNet/CNN) is a deep learning network that has superior performance with image inputs compared to regular neural networks. Before CNNs were created, manual feature extraction methods were used that did not perform anywhere near as well and were very time consuming.
<p align="center">
    <img src="Images/cnn.jpeg" alt="convolutional neural network" style="max-width:600px;"/>
</p>
![convolutional neural network](Images/cnn.jpeg)

A basic CNN can be split up into three important layers: the convolutional layer, the pooling layer and the fully-connected layer. All three will be explained in detail below so you can understand how a CNN works at each step.

## Convolutional Layer
The convolutional layer is the heart of the CNN architeture, it is where to majority of computations occur. To imagine how a convolutional layer works it is best to imagine a flashlight shining over the top left of the input image. In this imaginary example the flashlight covers a 5 x 5 area of a 32 x 32 image. This area is known as a <strong>receptive</strong> field and the flashlight is known as the <strong>filter</strong>. The filter is an array of numbers/weights and the dot product of the receptive field and filter is calculated and fed into the output array. Now for the whole image, the flashlight shines on a new section every time calculating the dot product, this is known as <strong>convolving</strong>. The amount of pixels the flashlight moves is known as the <strong>stride</strong>, it can be one or many pixels. When the filter has finished convolving over the whole image, the final output array is the <strong>feature map</strong>.

<p align="center">
    <img src="Images/convlayer.png" alt="convolutional layer" style="max-width:600px;"/>
</p>

