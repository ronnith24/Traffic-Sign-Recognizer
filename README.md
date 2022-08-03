# A CNN to recognize Traffic Signs

##  Model
In this architecture there are 2 convolutional layers followed by a pooling layer and then 2 convolution and pooling pairs as seen in the code. At the end there was a dense layer with a softmax activation, all other activations were the standard ReLU activation. I decided to train the model with the adam optimization algorithm and the binary crossentropy loss function. Later I added dropout to regularize the model.

### Performance:
> Time: Each step took approximately 70ms
> Training Accuracy: 0.9818
> Testing Accuracy: 0.9870
