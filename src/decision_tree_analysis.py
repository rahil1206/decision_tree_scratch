"""
    This module implements the decision tree learning algorithm and handles the data.
"""
import os
import sys

import numpy as np

# Global variable stores the number of unique labels in the dataset.
NUM_LABELS = -1


def import_dataset(path_to_dataset):
    """Imports dataset as numpy matrices."""
    dataset = np.loadtxt(path_to_dataset, dtype="float")
    # Set global variable to store number of labels in the dataset
    global NUM_LABELS
    NUM_LABELS = len(np.unique(dataset[:, dataset.shape[1] - 1].flatten()))
    return dataset


def get_features_and_labels(dataset):
    """Separates the given dataset into features and labels."""
    # Everything in the dataset numpy matrix is a feature except the last column.
    features = dataset[:, :dataset.shape[1] - 1]
    labels = dataset[:, dataset.shape[1] - 1:].flatten()
    return features, labels


def calculate_information_gain(all_frequencies, lbranch_frequencies, rbranch_frequencies):
    """Calculates the information gain. Each input parameter is a list containing frequencies for each attribute."""
    return calculate_h(all_frequencies) - calculate_remainder(lbranch_frequencies, rbranch_frequencies)


def calculate_h(frequencies):
    """Calculates the defined h function. The input parameter is a list containing frequencies of each label."""
    total_entries = sum(frequencies)
    output = 0
    # Iterate through the frequency of each attribute.
    for entry in frequencies:
        if entry == 0:
            continue
        pk = entry / total_entries
        output -= pk * np.log2(pk)
    return output


def calculate_remainder(lbranch_frequencies, rbranch_frequencies):
    """Calculates the defined remainder function. Each input parameter is a list containing frequencies
       for each label."""
    lbranch_number_of_entries = sum(lbranch_frequencies)
    rbranch_number_of_entries = sum(rbranch_frequencies)
    total_entries = lbranch_number_of_entries + rbranch_number_of_entries
    return (lbranch_number_of_entries * calculate_h(lbranch_frequencies) +
            rbranch_number_of_entries * calculate_h(rbranch_frequencies)) / total_entries


def find_split(training_dataset):
    """Implements the defined find_split function. The input parameter is a large numpy matrix representing the
       training dataset."""
    best_split = {"attribute": -1, "value": -1, "best_information_gain": -1}
    # Acquire the frequencies of each label in the training dataset as a 1-dimensional list.
    labels, total_frequencies = \
        np.unique(training_dataset[:, training_dataset.shape[1] - 1].flatten(), return_counts=True)
    # Iterate through each attribute in the dataset to find the best split.
    for attribute in range(training_dataset.shape[1] - 1):
        # Acquire the indices of the ordered dataset based on a particular attribute and set up the frequencies
        # of each label in the left and right branches to efficiently find the best split without moving around
        # records.
        ordered_attribute_indices = training_dataset[:, attribute].argsort()
        total_classes_lbranch = np.zeros(len(total_frequencies))
        total_classes_rbranch = total_frequencies.copy()
        i = 0
        attribute_analysed = False
        # After sorting the records based on an attribute, move the pivot of a potential split while keeping track
        # of the frequencies of the labels. The best split is found by finding the highest information gain on a
        # particular split for a particular attribute which will eventually replace the best_split variable.
        while i <= len(ordered_attribute_indices) - 2:
            # Infinite while loop considers multiple records with the same value for a particular attribute.
            while True:
                split_point = ordered_attribute_indices[i]
                next_split_point = ordered_attribute_indices[i + 1]
                # Adjust the frequencies of the labels on the left and right branches as the pivot of the
                # potential split moves.
                label = int(training_dataset[split_point][training_dataset.shape[1] - 1])
                index = np.where(labels == label)
                total_classes_lbranch[index] += 1
                total_classes_rbranch[index] -= 1
                i += 1
                # Break the loop if the next record in the ordered dataset does not have the same value as the previous
                # or if the entire dataset has been iterated. If the next record has the same value at the particular
                # attribute, then the pivot of the potential split needs to move passed it as the pivot cannot be in
                # between two idential values in that particular attribute.
                if training_dataset[split_point][attribute] != training_dataset[next_split_point][attribute]:
                    break
                elif i > len(ordered_attribute_indices) - 2:
                    # It means that the final two data have same value and we dont need to split
                    attribute_analysed = True
                    break
            if attribute_analysed:
                break

            # Calculate the information gain and update the stored best split if needed.
            information_gain = calculate_information_gain(total_frequencies,
                                                          total_classes_lbranch, total_classes_rbranch)
            if information_gain > best_split['best_information_gain']:
                best_split = {"attribute": attribute,
                              "value": (training_dataset[split_point][attribute] +
                                        training_dataset[next_split_point][attribute]) / 2.,
                              "best_information_gain": information_gain}

    return best_split


def create_default_node(attribute=-1, value=-1, left=None, right=None):
    """Creates a node with default values for the tree."""
    return {"attribute": attribute, "value": value, "left": left, "right": right}


def is_leaf_node(node):
    """Checks if the passed node is a leaf node."""
    return node["attribute"] == -1 and node["left"] is None and node["right"] is None


def split_dataset(dataset, best_split):
    """Splits the passed dataset based on the provided split information."""
    condition = dataset[:, best_split["attribute"]] < best_split["value"]
    lbranch = dataset[condition]
    rbranch = dataset[~condition]
    return lbranch, rbranch


def decision_tree_learning(training_dataset, depth=0):
    """Implements the defined decision tree learning algorithm."""
    features, labels = get_features_and_labels(training_dataset)
    if len(set(labels)) == 1:  # Base case when all the labels in the training dataset are the same.
        return create_default_node(value=labels[0]), depth
    else:  # Depth-first recursive step.
        best_split = find_split(training_dataset)
        ldataset, rdataset = split_dataset(training_dataset, best_split)
        lbranch, ldepth = decision_tree_learning(ldataset, depth + 1)
        rbranch, rdepth = decision_tree_learning(rdataset, depth + 1)
        node = create_default_node(attribute=best_split["attribute"],
                                   value=best_split["value"], left=lbranch, right=rbranch)
    return node, max(ldepth, rdepth)


def traverse_tree(trained_tree, data_point):
    """Traverses the tree to predict the label of the provided data_point."""
    node = trained_tree
    while not is_leaf_node(node):
        if data_point[node["attribute"]] < node["value"]:
            node = node["left"]  # Traverse into the left branch.
        else:
            node = node["right"]  # Traverse into the right branch.
    return int(node["value"])


def calculate_confusion_matrix(test_db, trained_tree):
    """Calculates the confusion matrix given a test dataset and a trained tree."""
    confusion_matrix = np.zeros((NUM_LABELS, NUM_LABELS))
    # Iterate through each data point in the test dataset.
    for test_data_point in test_db:
        predicted_label = traverse_tree(trained_tree, test_data_point)
        actual_label = int(test_data_point[len(test_data_point) - 1])
        if predicted_label == actual_label:  # Correct prediction.
            confusion_matrix[predicted_label - 1][predicted_label - 1] += 1
        else:  # Incorrect prediction.
            confusion_matrix[actual_label - 1][predicted_label - 1] += 1
    return confusion_matrix


def calculate_recall(confusion_matrix):
    """Calculates the recall per class using the confusion matrix."""
    recall = [confusion_matrix[i][i] / np.sum(confusion_matrix[i]) for i in range(NUM_LABELS)]
    return recall


def calculate_precision(confusion_matrix):
    """Calculates the precision per class using the confusion matrix."""
    precision = [confusion_matrix[i][i] / np.sum(confusion_matrix[:, i], axis=0) for i in range(NUM_LABELS)]
    return precision


def calculate_f1_measure(confusion_matrix):
    """Calculates the f1 measure per class using the confusion matrix."""
    recall = calculate_recall(confusion_matrix)
    precision = calculate_precision(confusion_matrix)
    f1_measure = [2 * recall[i] * precision[i] / (precision[i] + recall[i]) for i in range(NUM_LABELS)]
    return f1_measure


def evaluate(test_db, trained_tree):
    """Calculates the accuracy given a test dataset and a trained tree"""
    correct = 0
    # Iterate through each data point in the test dataset.
    for test_data_point in test_db:
        predicted_label = traverse_tree(trained_tree, test_data_point)
        actual_label = int(test_data_point[len(test_data_point) - 1])
        if predicted_label == actual_label:  # Correct prediction.
            correct += 1
    return correct / len(test_db)


def evaluate_with_confusion_matrix(confusion_matrix):
    """Calculates the accuracy using the confusion matrix."""
    total_data_points = np.sum(confusion_matrix)
    accuracy = sum([confusion_matrix[i][i] / total_data_points for i in range(NUM_LABELS)])
    return accuracy


def find_max_depth(node, depth=0):
    """Recursively finds the maximum depth of a given tree."""
    if is_leaf_node(node):  # Base case when a leaf node is traversed.
        return depth
    # Depth-first recursive step.
    return max(find_max_depth(node["left"], depth + 1), find_max_depth(node["right"], depth + 1))


def display_progress_bar(i, total_tests, size_of_progress_bar=30):
    """Utility function for displaying the progress bar when calculating the metrics."""
    completed_tests = int(size_of_progress_bar * i / total_tests)
    print("{}[{}{}] {}/{}".format("Calculating Performance Metrics: ", "#" * completed_tests,
                                  "." * (size_of_progress_bar - completed_tests), i, total_tests), end='\r')


def cross_validation(dataset, decision_tree_learning_algorithm, k=10):
    """Implements the k-fold cross validation algorithm and returns the performance metrics."""
    dataset_clone = dataset.copy()
    np.random.shuffle(dataset_clone)
    confusion_matrices = []
    max_tree_depths = []
    # Calculate the fold sizes as a list to split the dataset into: [X, X, X, ..., X, Y] where Y <= X.
    typical_fold_size = len(dataset) // k + (0 if len(dataset) % k == 0 else 1)
    test_fold_sizes = [typical_fold_size for _ in range(k - 1)]
    test_fold_sizes.append(len(dataset) - (k - 1) * typical_fold_size)
    index_position = 0
    for i in range(k):  # Each fold will be the test dataset and the rest will be the training dataset.
        display_progress_bar(i, k)  # Display a progress bar to keep the user notified.
        starting_index = index_position
        ending_index = index_position + test_fold_sizes[i]
        training_dataset = np.concatenate((dataset_clone[:starting_index], dataset_clone[ending_index:]))
        testing_dataset = dataset_clone[starting_index:ending_index]
        tree, _ = decision_tree_learning_algorithm(training_dataset)
        confusion_matrices.append(calculate_confusion_matrix(testing_dataset, tree))
        max_tree_depths.append(find_max_depth(tree))
        index_position = ending_index
    display_progress_bar(k, k)
    print("\n")
    # Calculate the test fold weights just in case the folds are uneven.
    test_fold_weights = np.array([i / sum(test_fold_sizes) for i in test_fold_sizes])
    averaged_confusion_matrix = sum([confusion_matrices[i] * test_fold_weights[i] for i in range(k)])
    return np.round(averaged_confusion_matrix, 2), \
        np.round(evaluate_with_confusion_matrix(averaged_confusion_matrix), 3), \
        np.round(calculate_precision(averaged_confusion_matrix), 3), \
        np.round(calculate_recall(averaged_confusion_matrix), 3), \
        np.round(calculate_f1_measure(averaged_confusion_matrix), 3), \
        np.round(np.mean(max_tree_depths), 2)


def prune(node, training_dataset, validation_dataset):
    """Prunes leaf nodes from the tree that do not worsen the model when absent."""
    if is_leaf_node(node["left"]) and is_leaf_node(node["right"]):
        # Base case where we are at a node with two leaf children nodes.
        lbranch_validation, rbranch_validation = split_dataset(validation_dataset, node)
        labels_training, total_frequencies_training = \
            np.unique(training_dataset[:, training_dataset.shape[1] - 1].flatten(), return_counts=True)
        majority_label = labels_training[np.argmax(total_frequencies_training)]
        new_node = create_default_node(value=majority_label)
        # If there is no data in the validation dataset at this node, then the model
        # has not worsened when the leaf nodes are absent so the leaf nodes are pruned.
        if len(lbranch_validation) == 0 or len(rbranch_validation) == 0:
            return new_node
        # Else use the validation dataset to see if the model worsened if the leaf nodes
        # are absent and prune accordingly.
        current_accuracy = evaluate(validation_dataset, node)
        new_accuracy = evaluate(validation_dataset, new_node)
        if new_accuracy >= current_accuracy:
            return new_node
        return node
    else:  # Depth-first recursive step to traverse the tree until we find a node with two leaf children nodes.
        lbranch_training, rbranch_training = split_dataset(training_dataset, node)
        lbranch_validation, rbranch_validation = split_dataset(validation_dataset, node)
        if not is_leaf_node(node["left"]):  # Recurse with the left branch if left child is not a leaf.
            node["left"] = prune(node["left"], lbranch_training, lbranch_validation)
        if not is_leaf_node(node["right"]):  # Recurse with the right branch if right child is not a leaf.
            node["right"] = prune(node["right"], rbranch_training, rbranch_validation)
        if is_leaf_node(node["left"]) and is_leaf_node(node["right"]):  # Just in case pruning occurred above.
            node = prune(node, training_dataset, validation_dataset)
        return node


def nested_cross_validation(dataset, decision_tree_learning_algorithm, prune_algorithm, k=10):
    """Implements the nested k-fold cross validation algorithm and returns the performance metrics."""
    dataset_clone = dataset.copy()
    np.random.shuffle(dataset_clone)
    confusion_matrices = []
    max_tree_depths_post_pruning = []
    # Calculate the fold sizes as a list to split the dataset into: [X, X, X, ..., X, Y] where Y <= X.
    typical_fold_size = len(dataset) // k + (0 if len(dataset) % k == 0 else 1)
    test_fold_sizes = [typical_fold_size for _ in range(k - 1)]
    test_fold_sizes.append(len(dataset) - (k - 1) * typical_fold_size)
    index_position = 0
    for i in range(k):  # Each fold will be the test dataset and the rest will be the training+validation datasets.
        starting_index = index_position
        ending_index = index_position + test_fold_sizes[i]
        testing_dataset = dataset_clone[starting_index:ending_index]
        validation_and_training_dataset = np.concatenate((dataset_clone[:starting_index], dataset_clone[ending_index:]))
        index_position_nested_folds = 0
        validation_and_training_fold_sizes = test_fold_sizes.copy()
        del validation_and_training_fold_sizes[i]  # Fold sizes without the test dataset's fold.
        for j in range(k - 1):  # Iterate through other folds cycling as the validation dataset,
            # the rest is used for training.
            display_progress_bar(i * (k - 1) + j, k * (k - 1))  # Display a progress bar to keep the user notified.
            starting_index_nested_folds = index_position_nested_folds
            ending_index_nested_folds = index_position_nested_folds + validation_and_training_fold_sizes[j]
            validation_dataset = validation_and_training_dataset[starting_index_nested_folds:ending_index_nested_folds]
            training_dataset = np.concatenate((validation_and_training_dataset[:starting_index_nested_folds],
                                               validation_and_training_dataset[ending_index_nested_folds:]))
            tree, _ = decision_tree_learning_algorithm(training_dataset)
            prune_algorithm(tree, training_dataset, validation_dataset)
            max_tree_depths_post_pruning.append(find_max_depth(tree))
            confusion_matrices.append(calculate_confusion_matrix(testing_dataset, tree))
            index_position_nested_folds = ending_index_nested_folds
        index_position = ending_index
    display_progress_bar(k * (k - 1), k * (k - 1))
    print("\n")
    # Calculate the test fold weights just in case the folds are uneven.
    test_fold_weights = np.array([i / (sum(test_fold_sizes) * (k - 1)) for _ in range(k - 1) for i in test_fold_sizes])
    averaged_confusion_matrix = sum([confusion_matrices[i] * test_fold_weights[i] for i in range(k * (k - 1))])
    return np.round(averaged_confusion_matrix, 2), \
        np.round(evaluate_with_confusion_matrix(averaged_confusion_matrix), 3), \
        np.round(calculate_precision(averaged_confusion_matrix), 3), \
        np.round(calculate_recall(averaged_confusion_matrix), 3), \
        np.round(calculate_f1_measure(averaged_confusion_matrix), 3), \
        np.round(np.mean(max_tree_depths_post_pruning), 2)


def parse_paths_to_datasets(paths):
    """Converts a string representation of a list of paths to datasets into a list object."""
    if "'" in paths:
        path_as_list = paths.split("'")  # Contains many datasets.
    else:
        return [paths]  # Contains one dataset.
    # Remove elements that are not paths.
    if path_as_list[0].strip() == "[":
        del path_as_list[0]
    if path_as_list[-1].strip() == "]":
        del path_as_list[-1]
    for item in path_as_list:
        if item.strip() == "," or item.strip() == "":
            path_as_list.remove(item)
    return path_as_list


def run():
    """Runs the script."""
    # Capture and parse user input.
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        print("usage: python3 -m decision_tree_analysis paths_to_datasets [random_seed]")
        sys.exit(0)
    if len(sys.argv) == 3:
        np.random.seed(int(sys.argv[2]))

    # Iterate through each provided dataset to run cross validation and nested cross validation.
    for path_to_dataset in parse_paths_to_datasets(sys.argv[1]):
        dataset = import_dataset(path_to_dataset)
        # Cross Validation.
        print(f"\nCross Validation Algorithm Running on {os.path.basename(path_to_dataset)}")
        confusion_matrix, accuracy, precision, recall, f1_measure, max_tree_depth = \
            cross_validation(dataset, decision_tree_learning)
        print(f"Average Confusion Matrix:\n{confusion_matrix}")
        print(f"Average Accuracy: {accuracy}")
        print(f"Average Precision per Class: {precision}")
        print(f"Average Recall per Class: {recall}")
        print(f"Average F1 Measure per Class: {f1_measure}")
        print(f"Average Maximum Tree Depth: {max_tree_depth}")
        # Nested Cross Validation with pruning.
        print(f"\nNested Cross Validation Algorithm Running on {os.path.basename(path_to_dataset)}")
        confusion_matrix, accuracy, precision, recall, f1_measure, max_tree_depth = \
            nested_cross_validation(dataset, decision_tree_learning, prune)
        print(f"Average Confusion Matrix:\n{confusion_matrix}")
        print(f"Average Accuracy: {accuracy}")
        print(f"Average Precision per Class: {precision}")
        print(f"Average Recall per Class: {recall}")
        print(f"Average F1 Measure per Class: {f1_measure}")
        print(f"Average Maximum Tree Depth: {max_tree_depth}")


if __name__ == "__main__":
    run()
