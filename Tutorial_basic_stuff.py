'''
Normalize Input:
Always have variables to have zero-mean and equal variance (normalization)

Random Initialization (initial weight and bias):
random_normal (use zero_mean, small std)

Stochastic Gradient Descent:
compute not loss, but estimation of loss:
average of loss of randomly picked small sections of data (1000 samples)
very small step lengths
Input: zero-mean, small variance
initialize weight with zero mean, equal variance small

Using Momentum instead of -alpha*gradient
M = 0.9*M + delta(loss)
this is to use our previous knowledge of descending direction

Learning Rate Decay...

NN Structure:
W*input + b (linear) -> ReLu/Logistic (Non-Linear) -> W2*ReLu + b -> softmax (classfication) -> cost function

ConvNets:
For each convolution layer, we use kernel(filter) to make the the feature map smaller (valid padding), and increase the
depth (# of layers), however, by doing so, we lose a lot of info
Thus, using pooling, using same padding or 1 striding, and max pooling or average pooling to reduce the size of feature
map but without losing info
structure:
image->conv layer->max pooling->conv layer->max pooling->fully connected->fully connected->classifier
'''