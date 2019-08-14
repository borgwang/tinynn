## NN painting
Mimic a picture using a fully-connected neural. The input to the network are the coordinate of pixels of the image i.e. (x, y) and the ground true label is the corresponding RGB of that pixel.

Basically it's a regression task which neural network tries to learn the mapping between coordinate and RGB value of a specify picture. We use the [Square loss](https://en.wikipedia.org/wiki/Mean_squared_error) which leads to some kind of blurry effect.

## Example  
Origin picture  
<img src="https://github.com/borgwang/toys/raw/master/nn_paint/res/origin.jpg" width = "256" height = "160" alt="origin" align=center />  

Painting  
<img src="https://media.giphy.com/media/9xjTwYDV6zIW1aE62R/giphy.gif" width = "256" height = "160" alt="paint" align=center />   

## Architecture
We use a 4 layers full-connected neural network and train it with Adam optimizer(learning_rate=0.0003).  
Feel free to modify the code to build your own painter.  
