import numpy as np

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################
                a = X[i, :]
                b = self.X_train[j, :]
                dists[i, j] = np.sum(np.square(np.subtract(a, b)))
                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################
            dists[i, :] = np.sum(np.square(self.X_train-X[i, :]), axis=1)
            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        t = np.sum(X ** 2, axis=1)
        t = np.reshape(t, (num_test, 1))  # t(500, 1)
        f = np.sum(self.X_train ** 2, axis=1).T
        # np.tile() 重复A，B次，这里的B可以时int类型也可以是元组类型。
        f = np.tile(f, (num_test, 1))  # f(500, 5000)
        ft = X.dot(self.X_train.T)
        print(t.shape, f.shape, ft.shape, X.shape, self.X_train.shape)
        dists = t + f - 2 * ft
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            u, indices = np.unique(closest_y, return_inverse=True)
            y_pred[i] = u[np.argmax(np.bincount(indices))]
            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################

        return y_pred

# np.unique() # Find the unique elements of an array.
# >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
# >>> u, indices = np.unique(a, return_index=True)
# >>> u
# array(['a', 'b', 'c'],
#        dtype='|S1')
# >>> indices
# array([0, 1, 3])
# >>> a[indices]
# array(['a', 'b', 'c'],
#        dtype='|S1')

# return_inverse : bool, optional
# If True, also return the indices of the unique array (for the specified axis, if provided) that can be
# used to reconstruct ar.
# >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
# >>> u, indices = np.unique(a, return_inverse=True)
# >>> u
# array([1, 2, 3, 4, 6])
# >>> indices
# array([0, 1, 4, 3, 1, 2, 1])
# >>> u[indices]
# array([1, 2, 6, 4, 2, 3, 2])

# np.bincount() # Count number of occurrences of each value in array of non-negative ints.
# >>> np.bincount(np.arange(5))
# array([1, 1, 1, 1, 1])
# >>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
# array([1, 3, 1, 1, 0, 0, 0, 1])
