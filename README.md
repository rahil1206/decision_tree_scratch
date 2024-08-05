# Decision Tree Learning for WiFi Location

This project implements a decision tree algorithm from scratch to determine indoor locations based on WiFi signal strengths. Developed as part of a group for the Introduction to Machine Learning coursework at Imperial College London.

## Project Structure

- `src/`: Contains the source code
  - `decision_tree_analysis.py`: Main script for cross-validation and performance metrics
  - `decision_tree_visualisation.py`: Script to generate decision tree visualisations
- `data/`: Directory for dataset files
- `requirements.txt`: List of Python package dependencies
- `Report.pdf`: A brief report on the analysis

## Setup

### Prerequisites

- Python 3.10
- pip (Python package installer)

### Installation

1. Clone this repository:
   ```git clone https://github.com/rahil1206/decision_tree_scratch.git```
2. Navigate to the project directory:
  ```cd decision_tree_scratch```
3. Install required packages:
  ```pip install -r requirements.txt```

## Usage

### Decision Tree Analysis

Run the `decision_tree_analysis.py` script from the `src` directory to perform cross-validation and generate performance metrics.

Syntax:
```python3 -m decision_tree_analysis paths_to_datasets [random_seed]```

Examples:
```
python3 -m decision_tree_analysis "../data/clean_dataset.txt"
python3 -m decision_tree_analysis "../data/clean_dataset.txt" 123
python3 -m decision_tree_analysis "['../data/clean_dataset.txt', '../data/noisy_dataset.txt']"
python3 -m decision_tree_analysis ['../data/clean_dataset.txt','../data/noisy_dataset.txt'] 567
```
Note: Avoid using single quotes (') in file or directory names.

### Decision Tree Visualisation

Run the `decision_tree_visualisation.py` script from the `src` directory to generate a PNG image of the decision tree.

Syntax:
```python3 -m decision_tree_visualisation path_to_dataset```

Example:
```python3 -m decision_tree_visualisation "../data/clean_dataset.txt"```

## Features

- Implementation of decision tree learning algorithm
- Handling of continuous attributes and multiple labels
- 10-fold cross-validation for performance evaluation
- Nested cross-validation for pruned tree evaluation
- Decision tree visualisation

## Datasets

The project uses two datasets:
1. Clean dataset: `clean_dataset.txt`
2. Noisy dataset: `noisy_dataset.txt`

Each dataset contains 2000 samples with 7 WiFi signal strengths and a room number label.

## Output

The analysis script generates various performance metrics, including:
- Confusion matrices
- Accuracy scores
- Recall and precision rates
- F1-measures

The visualisation script produces a PNG image of the generated decision tree.

## Notes

- Ensure all file paths are correctly specified when running the scripts.
- The random seed parameter in the analysis script is optional but can be used for reproducibility.

With this implementation our group was able to achieve full marks on this coursework.