A pure `numpy` and `tensorflow` implementation of a Neural Network for classifying the MNIST digits. 

The numpy version gets to ~95% accuracy with just one hidden layer and 30 neurons. The TensorFlow version gets to 98% with two hidden layers (300 and 100 neurons each) and dropout layers. 

I've also included a small script to take an image of a number as a `.png` file (from e.g Paint) and convert it to a form similar to the training data. You can then get your network to classify your own writing! This is in `numpy_version/process_digit.py`

