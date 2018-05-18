import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import scipy.ndimage
from tensorflow.examples.tutorials.mnist import input_data

class network(object):
    def __init__(self):
        tf.reset_default_graph()
        self.test_batch_size = 256
        self.train_batch_size = 64
        self.total_iterations = 0

    def setup(self,load = None, structure=None,end_relu = False,end_biases = False, data = None, offset = 0):
        """
        Creates a network
        load: the filepath of the network to load (must be compatible with "structure"), if none then a new network will be created
        structure: an array determining the type of network and the hidden structure of the network. The array has the shape:
            [size of filters/number of nodes, number of filters/MLP, use pooling/use_relu, use biases];
            If the second input is <=0 then the network will be a MPL, else it will be a ConvNet.
            the last two inputs get converted to boolean from 1 or 0; they determine if the layer has biases and pooling/ReLUs on the layer. (convnets always use ReLUs). There is a fully connected layer added by default at the end of the network, this should not be in the structure array.
        end_relu: determines if the final layer has a ReLU;
        end_biases: determines if the final layer has biases;
        data: none defaults to the MNIST dataset from the Tensorflow examples folder, but others can be used (supplying the MNIST dataset is quicker if it is already loaded);
        offset: puts an offset on all images coming into the network e.g -0.5 will make all MNIST images between -0.5 and 0.5 instead of 0 to 1
        
        """
        self.offset = offset
        self.structure = structure
        self.data = data
        if (self.data is None):
            self.data = input_data.read_data_sets('data/MNIST/', one_hot=True)

        # We know that MNIST images are 28 pixels in each dimension.
        self.img_size = 28
        # Images are stored in one-dimensional arrays of this length.
        self.img_size_flat = self.img_size * self.img_size
        # Tuple with height and width of images used to reshape arrays.
        self.img_shape = (self.img_size, self.img_size)
        # Number of colour channels for the images: 1 channel for gray-scale.
        self.num_channels = 1
        # Number of classes, one class for each of 10 digits.
        self.num_classes = 10

        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size_flat], name='x')
        self.offset_layer = tf.add(self.x,self.offset)
        self.x_image = tf.reshape(self.offset_layer, [-1, self.img_size, self.img_size, self.num_channels])
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, axis=1)

        self.layers = [tf.Tensor for i in range(self.structure.shape[0]+1)]
        self.weights = [tf.Variable for i in range(self.structure.shape[0]+1)]
        self.biases = [tf.Variable for i in range(self.structure.shape[0]+1)]

        i=0;
        while (i<self.structure.shape[0]):
            self.filter_size = self.structure[i,0]
            self.num_filters = self.structure[i,1]
            self.use_pooling = bool(self.structure[i,2])
            self.use_biases = bool(self.structure[i,3])
            if (self.num_filters>0):
                if (i==0):
                    self.layers[i],self.weights[i],self.biases[i] =\
                        self.new_conv_layer(input=self.x_image,
                                       num_input_channels=self.num_channels,
                                       filter_size=self.filter_size,
                                       num_filters=self.num_filters,
                                       use_pooling=self.use_pooling,
                                       use_biases =self.use_biases)
                else:
                    self.num_input_channels = self.structure[i-1,1]
                    self.layers[i],self.weights[i],self.biases[i] =\
                        self.new_conv_layer(input=self.layers[i-1],
                                       num_input_channels=self.num_input_channels,
                                       filter_size=self.filter_size,
                                       num_filters=self.num_filters,
                                       use_pooling=self.use_pooling,
                                       use_biases =self.use_biases)
                i=i+1

            else:
                if (i==0):
                    self.layer_flat, self.num_features = self.flatten_layer(self.x_image)
                    self.layers[i],self.weights[i],self.biases[i] = self.new_fc_layer(input=self.layer_flat,
                                                                         num_inputs=self.num_features,
                                                                         num_outputs=self.filter_size,
                                                                         use_relu=self.use_pooling,
                                                                         use_biases =self.use_biases)
                else:
                    if(self.structure[i-1,1]>0):
                        self.image_flat, self.num_pixels = self.flatten_layer(self.layers[i-1])
                        self.layers[i],self.weights[i],self.biases[i] =  self.new_fc_layer(input=self.image_flat,
                                                                               num_inputs=self.num_pixels,
                                                                               num_outputs=self.filter_size,
                                                                               use_relu=self.use_pooling,
                                                                               use_biases =self.use_biases)

                    else:
                        self.layers[i],self.weights[i],self.biases[i] =  self.new_fc_layer(input=self.layers[i-1],
                                                                           num_inputs=self.num_features,
                                                                           num_outputs=self.filter_size,
                                                                           use_relu=self.use_pooling,
                                                                           use_biases =self.use_biases)
                self.num_features = self.filter_size
                i=i+1

        if(self.structure[i-1,1]>0):
            self.layer_last,self.num_features = self.flatten_layer(self.layers[i-1])
        else:
            self.layer_last = self.layers[i-1]

        self.layers[i],self.weights[i],self.biases[i]= self.new_fc_layer(input=self.layer_last,
                                                                         num_inputs=self.num_features,
                                                                         num_outputs=self.num_classes,
                                                                         use_relu=end_relu,
                                                                         use_biases = end_biases)

        self.y_pred = tf.nn.softmax(self.layers[i])

        self.y_pred_cls = tf.argmax(self.y_pred, axis=1)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.layers[i],
                                                                labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        if (load is not None):
            if load=="default":
                load ="./Models/MNIST_model"
            self.session = tf.Session()
            self.saver = tf.train.Saver()
            self.saver.restore(self.session,load)

    def save(self,location = "./Models/MNIST_model"):
        """
        saves the network at the given file path, defaults to "./Models/MNIST_model".
        """
        self.saver = tf.train.Saver()
        self.saver.save(self.session, location)

    def plot_images(self,images, cls_true, cls_pred=None):
        """
        plots 9 supplied images in a 3x3 grid, together with the true and predicted class labels. 
        """
        assert len(images) == len(cls_true) == 9

        # Create figure with 3x3 sub-plots.
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Plot image.
            ax.imshow(images[i].reshape(self.img_shape), cmap='binary')

            # Show true and predicted classes.
            if self.cls_pred is None:
                xlabel = "True: {0}".format(self.cls_true[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(self.cls_true[i], self.cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def new_weights(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self,length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def new_conv_layer(self,
                       input,         # The previous layer.
                       num_input_channels, # Num. channels in prev. layer.
                       filter_size,        # Width and height of each filter.
                       num_filters,        # Number of filters.
                       use_pooling=True,
                       use_biases =True
                       ):

        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        weights = self.new_weights(shape=shape)

        if(use_biases):
            # Create new biases, one for each filter.
            biases = self.new_biases(length=num_filters)
        else:
            biases = []

        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        if(use_biases):
            layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)

        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights, biases

    def flatten_layer(self,layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features

    def new_fc_layer(self,input,          # The previous layer.
                     num_inputs,     # Num. inputs from prev. layer.
                     num_outputs,    # Num. outputs.
                     use_relu=True,
                     use_biases = True): # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        if (use_biases):
            biases = self.new_biases(length=num_outputs)
        else:
            biases =[]
        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        if (use_biases):
            layer = tf.matmul(input, weights) + biases
        else:
            layer = tf.matmul(input, weights)
        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer,weights,biases

    def optimize(self,num_iterations):

        # Start-time used for printing time-usage below.
        start_time = time.time()

        for i in range(self.total_iterations,
                       self.total_iterations + num_iterations):

            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = self.data.train.next_batch(self.train_batch_size)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {self.x: x_batch,
                               self.y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            self.session.run(self.optimizer, feed_dict=feed_dict_train)

            # Print status every 100 iterations.
            if i % 100 == 0:
                # Calculate the accuracy on the training-set.
                acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)

                # Message for printing.
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

                # Print it.
                print(msg.format(i + 1, acc))

        # Update the total number of iterations performed.
        self.total_iterations += num_iterations

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def plot_example_errors(self,cls_pred, correct):
        # This function is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # correct is a boolean array whether the predicted class
        # is equal to the true class for each image in the test-set.

        # Negate the boolean array.
        incorrect = (correct == False)

        # Get the images from the test-set that have been
        # incorrectly classified.
        images = self.data.test.images[incorrect]

        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = self.data.test.cls[incorrect]

        # Plot the first 9 images.
        plot_images(images=images[0:9],
                    cls_true=cls_true[0:9],
                    cls_pred=cls_pred[0:9])

    def plot_confusion_matrix(self,cls_pred):
        # This is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # Get the true classifications for the test-set.
        cls_true = self.data.test.cls

        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_true,
                              y_pred=cls_pred)

        # Print the confusion matrix as text.
        print(cm)

        # Plot the confusion matrix as an image.
        plt.matshow(cm)

        # Make various adjustments to the plot.
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, range(self.num_classes))
        plt.yticks(tick_marks, range(self.num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

        # Split the test-set into smaller batches of this size.

    def print_test_accuracy(self,show_example_errors=False,
                            show_confusion_matrix=False):
        # Number of images in the test-set.
        num_test = len(self.data.test.images)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_test:
            # The ending index for the next batch is denoted j.
            j = min(i + self.test_batch_size, num_test)

            # Get the images from the test-set between index i and j.
            images = self.data.test.images[i:j, :]

            # Get the associated labels.
            labels = self.data.test.labels[i:j, :]

            # Create a feed-dict with these images and labels.
            feed_dict = {self.x: images,
                         self.y_true: labels}

            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = self.session.run(y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        # Convenience variable for the true class-numbers of the test-set.
        cls_true = self.data.test.cls

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        # Calculate the number of correctly classified images.
        # When summing a boolean array, False means 0 and True means 1.
        correct_sum = correct.sum()

        # Classification accuracy is the number of correctly classified
        # images divided by the total number of images in the test-set.
        acc = float(correct_sum) / num_test

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_test))

        # Plot some examples of mis-classifications, if desired.
        if show_example_errors:
            print("Example errors:")
            plot_example_errors(cls_pred=cls_pred, correct=correct)

        # Plot the confusion matrix, if desired.
        if show_confusion_matrix:
            print("Confusion Matrix:")
            plot_confusion_matrix(cls_pred=cls_pred)

    def plot_conv_weights(self,weights, input_channel=0):
        # Assume weights are TensorFlow ops for 4-dim variables
        # e.g. weights_conv1 or weights_conv2.

        # Retrieve the values of the weight-variables from TensorFlow.
        # A feed-dict is not necessary because nothing is calculated.
        w = self.session.run(weights)

        # Get the lowest and highest values for the weights.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        w_min = np.min(w)
        w_max = np.max(w)

        # Number of filters used in the conv. layer.
        num_filters = w.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(num_grids, num_grids)

        # Plot all the filter-weights.
        for i, ax in enumerate(axes.flat):
            # Only plot the valid filter-weights.
            if i<num_filters:
                # Get the weights for the i'th filter of the input channel.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = w[:, :, input_channel, i]

                # Plot image.
                ax.imshow(img, vmin=w_min, vmax=w_max,
                          interpolation='nearest', cmap='seismic')

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def plot_conv_layer(self,layer, image):
        # Assume layer is a TensorFlow op that outputs a 4-dim tensor
        # which is the output of a convolutional layer,
        # e.g. layer_conv1 or layer_conv2.

        # Create a feed-dict containing just one image.
        # Note that we don't need to feed y_true because it is
        # not used in this calculation.
        feed_dict = {self.x: [image]}

        # Calculate and retrieve the output values of the layer
        # when inputting that image.
        values = self.session.run(layer, feed_dict=feed_dict)

        # Number of filters used in the conv. layer.
        num_filters = values.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(num_grids, num_grids)

        # Plot the output images of all the filters.
        for i, ax in enumerate(axes.flat):
            # Only plot the images for valid filters.
            if i<num_filters:
                # Get the output image of using the i'th filter.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = values[0, :, :, i]

                # Plot image.
                ax.imshow(img, interpolation='nearest', cmap='binary')

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def plot_image(self,image):
        plt.imshow(image.reshape(self.img_shape),
                   interpolation='nearest',
                   cmap='binary')

        plt.show()

    def average_output(self,layer, image,suppress_out = False):
        """
        layer: needs to be a tensorflow object, e.g net.layer[0]
        suppress_out determines if the image produced is displayed or not (bool)
        returns the average of a given layer’s outputs for a given image.
        """

        # Create a feed-dict containing just one image.
        # Note that we don't need to feed y_true because it is
        # not used in this calculation.
        feed_dict = {self.x: [image]}

        # Calculate and retrieve the output values of the layer
        # when inputting that image.
        values = self.session.run(layer, feed_dict=feed_dict)

        # Number of filters used in the conv. layer.
        num_filters = values.shape[3]

        img = values[0, :, :, 0]

        i=1
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = img + values[0, :, :, i]

        if suppress_out==False:
            img = img/num_filters
            plt.imshow(img)  
            plt.show()
        
        return img


    def give_prob(self,image,layer = None):
        """
        gives the network output at a given layer (default is final)
        """
        if(layer is None):
            output = self.y_pred
        else:
            output = self.layers[layer]

        img_shape=self.img_shape
        if image.ndim == 4:
            pre_class = np.zeros([image.shape[0],10])
            for i in range(0,image.shape[0]):
                im = image[i,:,:,0].reshape(img_shape[0]*img_shape[1])
                feed_dict = {self.x: [im],
                    self.y_true: [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]}
                pre_class[i,:] = self.session.run([output], feed_dict=feed_dict)[0]
            return pre_class
        elif image.ndim==3:
            im = image[:,:,0].reshape(img_shape[0]*img_shape[1])
        elif image.ndim==2:
            im = image[:,:].reshape(img_shape[0]*img_shape[1])

        elif image.ndim==1:
            im = image
        feed_dict = {self.x: [im],
                    self.y_true: [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]}
        #y_true is unknown for this image so the vector can be anything
        # Calculate the predicted class using TensorFlow.
        return self.session.run([output], feed_dict=feed_dict)[0]

    def give_class(self,image):
        img_shape=self.img_shape
        if image.ndim == 4:
            pre_class = np.zeros([image.shape[0],10])
            for i in range(0,image.shape[0]):
                im = image[i,:,:,0].reshape(img_shape[0]*img_shape[1])
                feed_dict = {self.x: [im],
                    self.y_true: [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]}
                pre_class[i,:] = self.session.run([self.y_pred], feed_dict=feed_dict)[0]
            return pre_class
        elif image.ndim==3:
            im = image[:,:,0].reshape(img_shape[0]*img_shape[1])
        elif image.ndim==2:
            im = image[:,:].reshape(img_shape[0]*img_shape[1])

        elif image.ndim==1:
            im = image
        feed_dict = {self.x: [im],
                    self.y_true: [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]}
        #y_true is unknown for this image so the vector can be anything
        # Calculate the predicted class using TensorFlow.
        return self.session.run([self.y_pred_cls], feed_dict=feed_dict)[0][0]

    def find_plane(self,x,Weights=None,Biases = None,node=None,layer=None):
        """
        This function finds the W,B and y for which xW+B = y.
        When specified this is done only for the "node" at depth "layer" of the network.
        Else it is done for all nodes in the final layer.
        outputs W,B,y

        x is the input being analyised as a numpy array (flattened or not)
        weights is a python list containing the weight matrices for each layer [layer0_weights, layer1_weights....]
        biases is a python list containing the bias vectors for each layer[layer0_biases, layer1_biases....]
        node is either None, int or list with 2 elements.
            When None the weight output will have shape (output size) and the pias will have size (output size)
            if node is an int, it is the node in the final layer for which the decision plane is being sought,
                hence the shapes will be: (input size)x1 and 1, respectivly
            if node is a 2 element list the decision plane will be the boundary between element 0 and element 1 in the list
            this can be faster than generating new lists
        layer is either None, an int or a list length 2
            when layer is None the whole network is analysed and the node arguament applies to the final layer
            when layer is an int the network only up to that layer will be analysed and node applies to that layer.
            When layer is a list length 2 it gives the begin and end layer to analyse- in htis case x should be the input to the smaller numbered layer.
            this can be faster than generating new lists

        the network structure is: x = Y0:
        Y0 R0 W0 + B0=Y1
        Y1 R1 W1 + B1 = Y2
        ...

        Note ther ReLU step is on the input not the output, the node and layer are with reference to Y (hence layer cannot be 0).
        """

        #check valid layer to investigate
        nlayersmin=0
        if layer is not None:
            if np.isscalar(layer):
                nlayers = layer
            elif len(layer) == 2:
                nlayersmin = min(layer)
                nlayers = max(layer)
            else:
                raise ValueError('layer length/type incorrect')

        else:
            if Weights is None:
                if Biases is None:
                    nlayers = len(self.layers)
                else:
                    nlayers=len(Biases)
            else:
                if Biases is None:
                    nlayers=len(Weights)
                else:
                    nlayers=min(len(Weights),len(Biases))
                    if len(Weights)!=len(Beights):
                        raise ValueError('Weight and Bias arrays must be the same length')

        if Weights is None:
            weights = []
            for L in range (nlayersmin,nlayers):
                weights.append(self.session.run(self.weights[L]).copy())
        else:
            weights = Weights.copy()

        if Biases is None:
            biases = []
            for L in range (nlayersmin,nlayers):
                biases.append(self.session.run(self.biases[L]).copy())
        else:
            biases = Biases.copy()


        #focus on the specified node.
        if node is not None:
            if np.isscalar(node):
                weights[-1] =np.array(weights[-1][:,node])
                biases[-1] = np.array([biases[-1][node]])
            elif len(node) == 2:
                weights[-1] =np.array(weights[-1][:,node[0]]-weights[-1][:,node[1]])
                biases[-1] = np.array([biases[-1][node[0]]-biases[-1][node[1]]])
            else:
                raise ValueError('node argument of improper type/length')

        #begin by creating RelU matrices
        y = x.flatten().copy()+self.offset
        R = [np.identity(x.size)]
        for L in range (0,nlayers-nlayersmin):
            y = y.dot(R[L]).dot(weights[L])+biases[L]
            r = (1.0*(y>=0))
            R.append(np.diag(r))

        #combine the weights, bias and ReLU matrices to produce W and B matrices
        W = 1
        Btemp = biases.copy()
        for L in range (0,nlayers-nlayersmin):
            #print(W.shape)
            Btemp[-1-L] = (Btemp[-1-L]).dot(W)
            W= (R[-2-L]).dot(weights[-1-L]).dot(W)

        B = sum(Btemp)
        return W,B,y
