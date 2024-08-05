Introduction to Machine Learning | Decision Tree Coursework

----prerequisites----

Please use Python 3.10 and install the required python packages by running pip install -r requirements.txt in the submission directory.

----decision_tree_analysis.py----

decision_tree_analysis.py will run the cross validation algorithm and the nested cross validation algorithm for each provided dataset and print the corresponding performance metrics.

Please run the decision_tree_analysis.py script in the src directory.

USAGE: python3 -m decision_tree_analysis paths_to_datasets [random_seed]

paths_to_datasets accepts both relative and absolute file paths.

examples accepted:
python3 -m decision_tree_analysis "D:\Coursework Assignment\data\clean_dataset.txt"
python3 -m decision_tree_analysis ../data/clean_dataset.txt 123
python3 -m decision_tree_analysis "['../data/clean_dataset.txt', '../data/noisy_dataset.txt']"
python3 -m decision_tree_analysis "['D:\Coursework Assignment\data\clean_dataset.txt', 'D:\Coursework Assignment\data\noisy_dataset.txt']"
python3 -m decision_tree_analysis ['../data/clean_dataset.txt','../data/noisy_dataset.txt'] 567

Please do not have ' characters in file or directory names.

----decision_tree_visualisation.py----

decision_tree_visualisation.py will create a .png image of the decision tree generated from a provided dataset.

Please run the decision_tree_visualisation.py script in the src directory.

USAGE: python3 -m decision_tree_visualisation path_to_dataset

path_to_dataset accepts both relative and absolute file paths.

examples accepted:
python3 -m decision_tree_visualisation "D:\Coursework Assignment\data\clean_dataset.txt"
python3 -m decision_tree_visualisation ../data/clean_dataset.txt

Please do not have ' characters in file or directory names.
