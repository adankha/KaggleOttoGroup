import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import csv


def main():

    # Delimiter is comma by default, if it is something else, have delimiter = ' '
    train_file = pd.read_csv('train.csv', header=0)
    test_file = pd.read_csv('test.csv', header=0)



    # Holds the headers provided in your csv file (row 0)
    train_column_labels = list(train_file.columns.values)
    test_column_labels = list(test_file.columns.values)



    train_data_array = train_file.as_matrix()

    class_map = {
        'Class_1': 1,
        'Class_2': 2,
        'Class_3': 3,
        'Class_4': 4,
        'Class_5': 5,
        'Class_6': 6,
        'Class_7': 7,
        'Class_8': 8,
        'Class_9': 9,
    }

    # Grab the ids associated with the train_data file
    ids1 = [x[0] for x in train_data_array]

    # Grab the classifications (or targets) of the train data array
    classifications = [class_map[x[-1]] for x in train_data_array]
    print('Printing classifications/targets')
    print(classifications)

    print('Printing the IDs associated with each product')
    print(ids1)

    # Drop the column that has the id as we do not need to have the unique IDs for fitting our data
    train_data = train_file.drop('id', 1)
    print('The train_file')
    print(train_file)
    # Convert the file (after dropping the first column) into a matrix
    train_data = train_data._get_numeric_data().as_matrix()

    print('The train data')
    print(train_data)

    # Convert the test file into a matrix
    test_file_array = test_file.as_matrix()

    # Grab the IDs assosciated with the test file
    ids2 = [x[0] for x in test_file_array]

    # Drop the IDs to use for testing
    test_data = test_file.drop('id', 1)

    #
    test_data = test_data._get_numeric_data()
    # Convert the test file as a matrix
    test_data = test_data.as_matrix()


    # Use a classifier to fit and train your model
    clf = GradientBoostingClassifier()
    clf.fit(train_data, classifications)
    print(clf.feature_importances_)
    print('printing predictions')

    # Create your predictions after training using the train set
    predictions = clf.predict_proba(test_data)


    l = len(predictions)
    hl = len(class_map)

    print('THE PREDICTIONS:::::::')
    print(predictions)

    # Create a 2D matrix array to hold the probabilities of classifications of your predictions
    # Since we have 9 classes, we have 9 columns (1 for each class).
    final_result = [[0 for col in range(hl)] for row in range(l)]

    # Traverse through the number of rows/products
    for i in range(l):

        # Traverse through each number (or percentage in our case) of each classification and add it a final_result
        for j in range(9):
            final_result[i][j] = predictions[i][j]
        #final_result[i][predictions[i] - 1] = 1

        # Insert the id associated with each row (should be in parallel)
        final_result[i].insert(0, ids2[i])

    # Grab all the keys of classifications
    result_header = list(class_map.keys())

    # re-insert the ID column ( as per request of the write-up in the kaggle project
    result_header.insert(0, 'id')

    # Add to the first row as result_header is our header of our csv file
    final_result.insert(0, result_header)

    l = len(final_result)
    with open('results.csv', 'w', newline='') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')
        for i in range(l):
            file_writer.writerow(final_result[i])


if __name__ == '__main__':
    main()
