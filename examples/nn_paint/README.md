Use a fully-connected neural network to mimic a image.  
The input to the network is the coordinate of pixels of the image i.e. (x, y) and the ground true label is the corresponding RGB of that pixel.

Basically it's a simple regression task which the neural network tries to learning the mapping between coordinate and RGB value of a specify image. The result shows some kind of 'smooth' effect.

## Example  
Origin picture
<img src="https://github.com/borgwang/toys/raw/master/nn_paint/res/origin.jpg" width = "256" height = "160" alt="origin" align=center />  

Painting  
<img src="https://media.giphy.com/media/9xjTwYDV6zIW1aE62R/giphy.gif" width = "256" height = "160" alt="paint" align=center />   

## Architecture
We use a 4 layers full-connected neural network and train it with Adam optimizer(learning_rate=0.0003).  

Feel free to modify the code to build your own painter.  
