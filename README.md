# Trees Fungal Disease Predictive Model

This project implements a predictive model for fungal disease classification in Penn State University (PSU) trees using machine learning techniques. The model is trained on tree feature data and uses a Histogram-based Gradient Boosting classifier to predict disease presence.

## Capabilities

- Loads training and test datasets from CSV files
- Handles missing values (`NA`) natively through the HistGradientBoostingClassifier
- Automatically converts features to numeric types
- Trains a robust gradient boosting model with the following hyperparameters:
  - `learning_rate`: 0.1
  - `max_iter`: 300
  - `max_depth`: None (unlimited)
  - `l2_regularization`: 0.0
  - `random_state`: 42
- Predicts disease classification for unseen test data
- Outputs predictions to a text file (`Predictions_HW3P3.txt`), one prediction per line

## Usage

### Prerequisites

- Python 3.x
- pandas
- numpy
- scikit-learn

### Installation

```bash
pip install pandas numpy scikit-learn
```

### Run Script

```bash
python3 FungalDisease_PM.py
```

The script will:
1. Read `DataTrain_HW3Problem3.csv` and `DataTest_HW3Problem3.csv`
2. Train the model on the training data
3. Generate predictions for the test data
4. Write predictions to `Predictions_HW3P3.txt`

## Use Cases

### Educational
This project serves as a practical example of applying supervised machine learning classification techniques to real-world environmental data. It demonstrates:
- Data preprocessing and cleaning (handling missing values, type conversion)
- Model selection and hyperparameter tuning for gradient boosting
- End-to-end workflow from raw data to predictions

### Environmental Monitoring
The predictive model can be adapted for:
- Early detection of fungal diseases in urban tree populations
- Monitoring tree health across campus or municipal landscapes
- Supporting arboricultural decision-making through data-driven insights

## Research Purposes

This project is designed for research purposes only. The repository includes dummy datasets for demonstration purposes. Penn State University (PSU), IST 557 Data Mining. Fall 2025.
