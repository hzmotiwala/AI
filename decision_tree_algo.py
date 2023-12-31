from __future__ import division

import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.

        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.

        Args:
            feature (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.

    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.

    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the provided data.

    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = None

    # TODO: finish this.
    #raise NotImplemented()

    a1 = DecisionNode(None, None, lambda feature: feature[0]==1)
    a2 = DecisionNode(None, None, lambda feature: feature[1]==1)
    a3 = DecisionNode(None, None, lambda feature: feature[2]==1)
    a4 = DecisionNode(None, None, lambda feature: feature[3]==1)

    decision_tree_root = a1
    
    a1.left = DecisionNode(None, None, None, 1)
    a1.right = a4

    a4.left = a2
    a4.right = a3

    a2.left = DecisionNode(None, None, None, 0)
    a2.right = DecisionNode(None, None, None, 1)

    a3.left = DecisionNode(None, None, None, 0)
    a3.right = DecisionNode(None, None, None, 1)

    #a2.left = DecisionNode(None, None, None, 1)
    #a2.right = DecisionNode(None, None, None, 0) 



    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.
    debug = False

    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0

    for i in range(len(classifier_output)):
        # true_positive or negative
        if classifier_output[i] == true_labels[i]:
            if true_labels[i] == 1:
                true_positive += 1
            else: 
                true_negative += 1
        # false pos or neg
        else:
            if classifier_output[i] == 1:
                false_positive += 1
            else:
                false_negative += 1

    if debug:
        print("tp: ", true_positive)
        print("tn: ", true_negative)
        print("fp: ", false_positive)
        print("fn: ", false_negative)



    return [[true_positive, false_negative],
         [false_positive, true_negative]]
    #raise NotImplemented()


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.

    Precision is measured as:
        true_positive/ (true_positive + false_positive)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.

    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0

    for i in range(len(classifier_output)):
        # true_positive or negative
        if classifier_output[i] == true_labels[i]:
            if true_labels[i] == 1:
                true_positive += 1
            else: 
                true_negative += 1
        # false pos or neg
        else:
            if classifier_output[i] == 1:
                false_positive += 1
            else:
                false_negative += 1

    precision = true_positive/ (true_positive + false_positive)

    return precision

    #raise NotImplemented()


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.

    Recall is measured as:
        true_positive/ (true_positive + false_negative)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The recall of the classifier output.
    """

    # TODO: finish this.
    #raise NotImplemented()

    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0

    for i in range(len(classifier_output)):
        # true_positive or negative
        if classifier_output[i] == true_labels[i]:
            if true_labels[i] == 1:
                true_positive += 1
            else: 
                true_negative += 1
        # false pos or neg
        else:
            if classifier_output[i] == 1:
                false_positive += 1
            else:
                false_negative += 1

    recall = true_positive/ (true_positive + false_negative)

    return recall


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.

    Accuracy is measured as:
        correct_classifications / total_number_examples

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The accuracy of the classifier output.
    """

    # TODO: finish this.
    #raise NotImplemented()

    total_number_examples = len(classifier_output)
    correct_classifications = 0

    for i in range(len(classifier_output)):
        # true_positive or negative
        if classifier_output[i] == true_labels[i]:
            correct_classifications += 1
    
    accuracy = correct_classifications / total_number_examples

    return accuracy


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.

    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.

    Returns:
        Floating point number representing the gini impurity.
    """
    #raise NotImplemented()
    
    debug = False

    counts = Counter(class_vector)
    P_Y = counts[1]
    P_N = counts[0]
    total = P_Y + P_N
    if total ==0:
        return 0.0
    P_Yes = P_Y/total
    P_No = 1 - P_Yes

    gini_impurity = P_Yes * (1-P_Yes)   +   P_No * (1-P_No) 

    if debug:
        print(counts)
        print(P_Y)
        print(P_N)
        print(total)
        print(gini_impurity)



    return gini_impurity

def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    #raise NotImplemented()

    debug = False

    

    if debug:
        print("\n****gini gain called****")
        print("previous_classes: ", previous_classes)
        print()
        print("current_classes: ", current_classes)
        print("previous_classes shape: ", np.shape(previous_classes))
        print()
        print("current_classes shape: ", np.shape(current_classes))

    previous_gain = gini_impurity(previous_classes)

    current_total_len = 0
    for this_list in current_classes:
        current_total_len += len(this_list)

    # calc weighted gain
    current_gain = 0
    for this_class in current_classes:
        current_gain += gini_impurity(this_class) * (len(this_class)/current_total_len)

    gini_gain = previous_gain - current_gain

    #print("Gini gain: ", gini_gain)
    return gini_gain


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        #raise NotImplemented()
        debug = False

        # base case 1...if only 1 class option then return classes[0]
        if debug:
            print("build tree")
            print("\n classes: ", classes)
            print("\n classes: ", len(classes))
            print("features array")
            print(np.array(features))
            print("feature len: ", len(features))
            print("feature len 0: ", len(features[0]))


        class_counter = Counter(classes)
        if debug:
            print("class counter: ", class_counter)
            print("len class counter", len(class_counter))
        if len(class_counter)==1:
            if debug:
                print("error now? :")
                print("classes 0: ", int(classes[0]))
            return DecisionNode(None, None, None, int(classes[0]))

        # base case 2...if reached depth limit then return vote of current class
        # how many 1 and how many 0. if 1 is greater return  1
        if depth == self.depth_limit:
            ones = class_counter[1]
            zeros = class_counter[0]
            if ones > zeros:
                vote = 1
            elif ones == zeros:
                vote = np.random.randint(2)
            else: 
                vote = 0
            return DecisionNode(None, None, None, vote)

        # split continuous variables using median....later changed to mean since median didn't pass
        feature_medians = np.mean(features, axis=0)
        if debug:
            print()
            print("medians: ")
            print(feature_medians)

        # creat a new feature set when compared to that median value
        new_features = []
        features_transpose = np.array(features).T
        for x in range(len(feature_medians)):
            #print("for loop:")
            #print(x)
            #print(feature_medians[x])
            this_feature = features_transpose[x] >= feature_medians[x]
            new_features.append(this_feature)
        #print("new featuers before array: ", new_features)
        new_features = np.array(new_features).T
        #print()
        #print("new featuers after array: \n", new_features)

        # calculate gini gain for each feature
        new_feature_length = len(new_features[0])
        gini_gains = []
        for i in range(new_feature_length):
            this_feature = new_features[:,i]

            # create false list and true list for each feature
            false_list = []
            for x in range(len(this_feature)):
                    if this_feature[x] == False:
                        false_list.append(classes[x])
            
            true_list = []
            for x in range(len(this_feature)):
                    if this_feature[x] == True:
                        true_list.append(classes[x])

            # combine into list of lists
            current_class = []
            current_class.append(false_list)
            current_class.append(true_list)

            # create previous class list
            previous_classes = list(classes)

            # get the gain and append to gini gains
            this_gain = gini_gain(previous_classes, current_class)
            gini_gains.append(this_gain)
        
        # normalize gains
        gini_sum = np.sum(gini_gains)
        gini_gains = gini_gains/gini_sum

        # get the index of the feature and the feature with the highest gain
        highest_gain_index = np.argmax(gini_gains)
        highest_gain_feature = new_features[:,highest_gain_index]

        # get the index values for the left branch and right branch
        right_index = [i for i, j in enumerate(highest_gain_feature) if j==0] 
        left_index = [i for i, j in enumerate(highest_gain_feature) if j==1]
        if debug:
            print("len zero idx: ", len(left_index))
            print("zero idx: ", left_index)
            print("len 1 idx: ", len(right_index))
            print("1 idx: ", right_index)

        # create features that will go left and right
        features_left = np.array([features[i] for i in left_index])
        features_left = np.array([l.tolist() for l in features_left])
        features_right = np.array([features[i] for i in right_index])
        features_right = np.array([l.tolist() for l in features_right])
        if debug:
            print("features left: ", features_left)
            print("len features left: ", len(features_left))
            print("features right: ", features_right)
            print("len features right: ", len(features_right))


        # create class list that will go left and right
        left_class = [classes[i] for i in left_index]
        right_class = [classes[i] for i in right_index]
        if debug:
            print("left class len: ", len(left_class))
            print("right class len: ", len(right_class))


        # add to depth counter
        depth += 1
        
        # recursive step
        left = self.__build_tree__(features_left, left_class, depth)
        right = self.__build_tree__(features_right, right_class, depth)

        return DecisionNode(left, right, lambda feature: feature[highest_gain_index] > feature_medians[highest_gain_index])



    def classify(self, features):
        """Use the fitted tree to classify a list of example features.

        Args:
            features (list(list(int)): List of features.

        Return:
            A list of class labels.
        """

        # TODO: finish this.
        class_labels = []

        for row in range(0, len(features)):
            decision = self.root.decide(features[row])
            class_labels.append(decision)

        return class_labels

def create_list(start,finish):
    lst=[]
    lst.append(start)
    iters = finish - start - 1
    for i in range(iters):
        this_item = start + i + 1
        lst.append(this_item)
    #print(lst)
    return lst


def generate_k_folds(dataset, k):
    """Split dataset into folds.

    Randomly split data into k equal subsets.

    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).

    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.

    Returns:
        List of folds.
    """

    # TODO: finish this.
    #raise NotImplemented()
    debug = False

    folds = []


    if debug:
        print("K: ", k)
        print("dataset: ")
        print(dataset)
        print("dataset classes: ")
        print(dataset[1])
        print("len dataset: ", len(dataset))
        print("len dataset 0: ", len(dataset[0]))
        print("shape features: ", dataset[0].shape)
        print("features type: ", type(dataset[0]))
        print("len classes: ", dataset[1])
        print("classes type: ", type(dataset[1]))
        print("shape type: ", dataset[1].shape)

    dataset_rows = len(dataset[0])
    test_rows_per_fold = dataset_rows/k
    test_rows_per_fold = int(test_rows_per_fold)
    if debug:
        print("test_rows_per_fold: ", test_rows_per_fold)

    # change dataset to 2d array
    features = dataset[0]
    y_labels = np.array(dataset[1])
    y_labels = y_labels.reshape(-1, 1)
    if debug:
        print("features: ", features)
        print(len(features), features.shape)
        print("y_labels: ", y_labels)
        print(len(y_labels), y_labels.shape)
    new_dataset = np.hstack((features, y_labels))
    #print("new_dataset: ", new_dataset)


    # get the folds
    if debug:
        print("new dataset type: ", type(new_dataset))
        print(new_dataset[0:1,0:1])
        print(sum(new_dataset))
    np.random.shuffle(new_dataset)
    if debug:
        print(new_dataset[0:1,0:1])
        print(sum(new_dataset))
        print("new dataset type: ", type(new_dataset))
    

    copy_dataset = new_dataset
    indexes_to_remove = create_list(0,test_rows_per_fold)
    #print("indexes_to_remove: ", indexes_to_remove)

    test_set = copy_dataset[:test_rows_per_fold,:]
    train_set = np.delete(copy_dataset, indexes_to_remove, axis=0)
    # print("train_set: ", train_set.shape)
    # print("test_set: ", test_set.shape)

    # print(new_dataset[137])
    # print(train_set[0])
    # print(test_set[0])

    # create the folds
    for i in range(k):
        #print("\n new iter: ", i)
        copy_dataset = new_dataset
        row_to_start = i * test_rows_per_fold
        row_to_end = row_to_start + test_rows_per_fold
        #print("start: ", row_to_start)
        #print("end: ", row_to_end)
        indexes_to_remove = create_list(row_to_start,row_to_end)

        test_set = copy_dataset[row_to_start:row_to_end,:]
        #print("column to delete from examples: ", len(test_set[0])-1)
        test_set_examples = np.delete(test_set, len(test_set[0])-1, axis=1)
        test_set_classes = test_set[:,-1]
        #print("shape test_set_examples: ", test_set_examples.shape)
        #print("shape test_set_classes: ", test_set_classes.shape)
        train_set = np.delete(copy_dataset, indexes_to_remove, axis=0)
        train_set_examples = np.delete(train_set, len(train_set[0])-1, axis=1)
        train_set_classes = train_set[:,-1]
        #print("train_set: ", train_set.shape)
        #print("test_set: ", test_set.shape)
        
        test_tuple = (test_set_examples, test_set_classes)
        train_tuple = (train_set_examples, train_set_classes)

        folds.append((train_tuple, test_tuple))

    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.

         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

        #added by zane
        #self.features_used_per_tree = []
        #self.DT = DecisionTree()


    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.

            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.
        #raise NotImplemented()

        DT = DecisionTree(depth_limit=5)

        debug = True

        for i in range(self.num_trees):
            # build features subset
            features_array = np.array(features)
            features_rows = len(features_array)
            features_cols = len(features_array[0])
            sample_rows = int(features_rows*self.example_subsample_rate)
            sample_cols = int(features_cols*self.attr_subsample_rate)
            sample_row_indexes = np.random.randint(features_rows, size=sample_rows).tolist()
            sample_col_indexes = np.random.choice(features_cols, size=sample_cols, replace=False).tolist()
            sample_subset_rows = features_array[sample_row_indexes,:]
            #sample_features = sample_subset_rows[:,sample_col_indexes].tolist()
            # zero out the columns that are not needed
            for i in sample_col_indexes:
                sample_subset_rows[:,i] = 0
            sample_features = sample_subset_rows.tolist()

            # build classes list
            classes_array = np.array(classes)
            sample_classes = classes_array[sample_row_indexes].tolist()

            # build a tree
            self.trees.append(DT.__build_tree__(sample_features, sample_classes))


            # save the columns used for the tree
            #self.features_used_per_tree.append(sample_col_indexes)



    def classify(self, features):
        """Classify a list of features based on the trained random forest.

        Args:
            features (list(list(int)): List of features.
        """

        # TODO: finish this.
        class_labels = []

        # iter over every row in features
        for row in features:
            #iter over every tree
            answer = []
            for i in range(self.num_trees):
                # get features for this tree
                #row_array = np.array(row)
                #features_this_tree = row_array[self.features_used_per_tree[i]].tolist()
                label_this_tree = self.trees[i].decide(row)
                answer.append(label_this_tree)
            vote_this_row = sum(answer)/self.num_trees
            vote = int(round(vote_this_row))
            class_labels.append(vote)

        return class_labels

        # for row in range(0, len(features)):
        #    decision = self.root.decide(features[row])
        #    class_labels.append(decision)


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, num_trees=50, depth_limit=5, example_subsample_rate=0.66,
                 attr_subsample_rate=0.66):
        """Create challenge classifier.

        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        #raise NotImplemented()

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate


    def fit(self, features, classes):
        """Build the underlying tree(s).

            Fit your model to the provided features.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.
        DT = DecisionTree(depth_limit=5)

        debug = True

        for i in range(self.num_trees):
            # build features subset
            features_array = np.array(features)
            features_rows = len(features_array)
            features_cols = len(features_array[0])
            sample_rows = int(features_rows*self.example_subsample_rate)
            sample_cols = int(features_cols*self.attr_subsample_rate)
            sample_row_indexes = np.random.randint(features_rows, size=sample_rows).tolist()
            sample_col_indexes = np.random.choice(features_cols, size=sample_cols, replace=False).tolist()
            sample_subset_rows = features_array[sample_row_indexes,:]
            #sample_features = sample_subset_rows[:,sample_col_indexes].tolist()
            # zero out the columns that are not needed
            for i in sample_col_indexes:
                sample_subset_rows[:,i] = 0
            sample_features = sample_subset_rows.tolist()

            # build classes list
            classes_array = np.array(classes)
            sample_classes = classes_array[sample_row_indexes].tolist()

            # build a tree
            self.trees.append(DT.__build_tree__(sample_features, sample_classes))

    def classify(self, features):
        """Classify a list of features.

        Classify each feature in features as either 0 or 1.

        Args:
            features (list(list(int)): List of features.

        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        class_labels = []

        # iter over every row in features
        for row in features:
            #iter over every tree
            answer = []
            for i in range(self.num_trees):
                # get features for this tree
                #row_array = np.array(row)
                #features_this_tree = row_array[self.features_used_per_tree[i]].tolist()
                label_this_tree = self.trees[i].decide(row)
                answer.append(label_this_tree)
            vote_this_row = sum(answer)/self.num_trees
            vote = int(round(vote_this_row))
            class_labels.append(vote)

        return class_labels


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Args:
            data: data to be added to array.

        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Bonnie time to beat: 0.09 seconds.

        Args:
            data: data to be sliced and summed.

        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        #raise NotImplemented()
        new_data = data * data + data
        return new_data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Args:
            data: data to be added to array.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Bonnie time to beat: 0.07 seconds

        Args:
            data: data to be sliced and summed.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        #raise NotImplemented()
        data_100 = data[:100,:]
        sum_100 = np.sum(data_100, axis = 1)
        max_index = np.argmax(sum_100)
        max_row_sum = sum_100[max_index]
        return (max_row_sum,max_index)


    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Bonnie time to beat: 15 seconds

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """


        #raise NotImplemented()
        debug = False
        data_flat = data[data>0].astype(int).flatten()
        if debug:
            print("data: ", data_flat)
            print("data type: ", type(data_flat))
            print("data len: ", len(data_flat))
        counters = Counter(data_flat)
        if debug:
            print("counter: ", dict(counters))
            print("items: ", counters.items())
        #print("data len [0]: ", len(data_flat[0]))
        return counters.items()


