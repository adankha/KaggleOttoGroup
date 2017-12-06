"""
The following program is used for a Kaggle Competition.
The Competition can be found here:
https://www.kaggle.com/c/otto-group-product-classification-challenge/leaderboard
"""


import pandas as pd
from sklearn.ensemble import *
import csv


def pre_process(train_file, test_file):
    """
    This function is the pre-processing phase where it will essentially separate the classifications/targets from
    the columns in the train csv file and remove the IDS from both the train and test files.

    :param train_file: Holds the raw train file
    :param test_file: Holds the raw test file
    :return: the train_data (ready to be used for the classifier),
             classifications/targets (for the train_data)
             the test_data (ready to be used for the classifier)
    """

    # Grab the classifications (or targets) of the train data array
    classifications = train_file['target']

    # Drop the column that has the id and target as we do not need to have the unique IDs or target for fitting our data
    train_data = train_file.drop(['id', 'target'], axis=1)

    # Convert the file (after dropping the first column) into a matrix
    # This is the final stage of the data. It can now be used to fit into our model/classifiers
    train_data = train_data.as_matrix()

    # Drop the IDs to use for testing
    test_data = test_file.drop('id', axis=1)

    # Convert the test file as a matrix
    test_data = test_data.as_matrix()

    return train_data, classifications, test_data


def post_process(ids, predictions, class_map):
    """
    The following function is the 'post processing stage' where it consists of taking the predicted values
    from the classifier and adds certain columns and/or rows so that it has the exact layout as the Kaggle sample
    provided.

    :param ids: Holds the IDs of the test set
    :param predictions: Holds the predictions of each product / ID
    :param class_map: Holds a mapping of all the classifications
    :return: The newly formed array after adding all columns and rows.
    """

    # Create a 2D matrix array to hold the probabilities of classifications of your predictions
    # Since we have 9 classes, we have 9 columns (1 for each class).
    pred_len = len(predictions)
    class_len = len(class_map)

    data = [[0 for col in range(class_len)] for row in range(pred_len)]

    # Traverse through the number of rows/products
    for i in range(pred_len):

        # Traverse through each number (or percentage in our case) of each classification and add it a final_result
        for j in range(class_len):
            data[i][j] = predictions[i][j]

        # Insert the id associated with each row (should be in parallel)
        data[i].insert(0, ids[i])

    # Grab all the keys of classifications
    result_header = list(class_map.keys())

    # re-insert the ID column (as per request of the write-up in the kaggle project)
    result_header.insert(0, 'id')

    # Add result_header to our first row as it is our header. [id, Class_1, Class_2, ..., Class_9]
    data.insert(0, result_header)

    return data


def write_to_csv(data, model):
    """
    This is the final step to the project. It will Write to the CSV file after the post-processing phase.
    It is a comma delimited csv file.

    :param data: The array that was returned from the post_process(..) function
    :param model: The current model/classifier used
    :return: None
    """

    result_len = len(data)
    with open(model + '.csv', 'w', newline='') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')
        for i in range(result_len):
            file_writer.writerow(data[i])

    print('Finished writing ', model, 'to a csv file.')


def main():

    # These are all the classifications associated with the dataset
    class_map = {
        'Class_1': 1, 'Class_2': 2, 'Class_3': 3,
        'Class_4': 4, 'Class_5': 5, 'Class_6': 6,
        'Class_7': 7, 'Class_8': 8, 'Class_9': 9,
    }

    # These are the models used to assess scores.
    models = {
        #'rfc': RandomForestClassifier(), # Score: 1.45390
        #'adaboost': AdaBoostClassifier(), # Score: 2.02382
        #'gradientboost: GradientBoostingClassifier() # 0.59382
        'best_gradientboost': GradientBoostingClassifier(min_samples_split=1200, min_samples_leaf=60,
                                                         max_depth=60, max_features=7, random_state=42), # Score: 0.45283
    }

    # The delimiter is comma by default, if it is something else, have delimiter = ' '
    train_file = pd.read_csv('train.csv', header=0)
    test_file = pd.read_csv('test.csv', header=0)

    # Pre-processing step:
    train_data, classifications, test_data = pre_process(train_file, test_file)

    # Grab the IDs associated with the test file
    ids_test = test_file['id']

    for model in models.keys():

        # Use a classifier to fit and train your model
        clf = models[model]

        # Fit the data to your classifier.
        clf.fit(train_data, classifications)

        # Create your predictions after training using the train set
        predictions = clf.predict_proba(test_data)

        # Post-process step:
        # The following function returns the predicted values for each row, with the correct layout
        # It mocks essentially what the sample excel spreadsheet looks like via the write-up
        final_result = post_process(ids_test, predictions, class_map)

        # Write to the csv file
        write_to_csv(final_result, model)


if __name__ == '__main__':
    main()
