import numpy as np 
import matplotlib.image as mpimg
import skimage.transform as T
import Neural_Net as N
import matplotlib.pyplot as plt 

#Any png image of a number
fname='my_handwriting.png'

#Read the image
img=mpimg.imread(fname)

#Turn it into the same form as the training data- so zeros everywhere apart from the line
digit=1.-np.median(img, axis=-1)

#Average sum of the training data images. Need to normalise to ths value
mean_test_data=103.0 #mean sum of each test image

#Resize the image down to 28x28 pixels
small_digit=T.resize(digit, (28, 28))
small_digit*=mean_test_data/np.sum(small_digit)
#And now turn it into the required format, a vector of shape 784, 1
small_digit_vector=small_digit.reshape(-1, 1)

#Load the best weights and biases
biases_1=np.load('biases_1.npy')
weights_1=np.load('weights_1.npy')
out_biases=np.load('biases_output.npy')
out_weights=np.load('weights_output.npy')

#Make a net without data, just for prediction
net=N.MNISTNetwork(saved_values=[weights_1, biases_1, out_weights, out_biases])
outputs=net.think(small_digit_vector)
prediction=np.argmax(outputs)
print 'This digit is: {}'.format(prediction)

plt.imshow(small_digit)
plt.title('This digit is: {}'.format(prediction))
plt.show()