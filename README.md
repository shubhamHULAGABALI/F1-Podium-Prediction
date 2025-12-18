# Formula 1 Podium Prediction Using Random Forest

## Project Overview
Formula 1 is a highly competitive and data-intensive motorsport where race outcomes are influenced by multiple interdependent factors, including driver performance, team strength, qualifying position, and circuit characteristics.
This project implements a Random Forest Classifier to predict whether a Formula 1 driver will finish on the podium (Top 3) using historical race data.

The objective of this project is to demonstrate the practical application of machine learning techniques in sports analytics through structured data preprocessing, feature engineering, and model evaluation.

---

## Problem Statement
Predicting podium finishes in Formula 1 is challenging due to:
- Strong class imbalance (only three podium positions per race)
- Non-linear relationships between performance variables
- Variability across seasons, teams, and circuits

This project addresses these challenges using an ensemble learning approach with balanced class handling.

---

## Dataset Description
The model is trained on historical Formula 1 data from 2022 to 2024, including:
- Race results
- Race metadata
- Constructor (team) information
- Driver statistics

All datasets are stored locally in the `Datasets/` directory.

---

## Feature Engineering
The following features were derived and used for training:

- Average Grid Position – Mean qualifying position per team
- Total Constructor Points – Overall team performance metric
- Average Finishing Position – Race consistency indicator
- Circuit ID – Encoded representation of track characteristics
- Driver Points – Individual driver performance
- Past Podiums – Historical podium count
- Races Participated – Driver experience level

---

## Model Architecture
- Algorithm: Random Forest Classifier
- Number of Estimators: 100
- Class Weight: Balanced
- Train-Test Split: 80% training, 20% testing
- Validation: 5-Fold Cross-Validation

Random Forest was selected due to its robustness against overfitting, ability to capture non-linear patterns, and suitability for structured tabular data.

---

## Model Evaluation
- Test Accuracy: ~85%
- Cross-Validation Accuracy: ~84%
- Strong performance in non-podium classification
- Reasonable podium detection despite inherent class imbalance

Balanced class weighting significantly improved podium recall compared to baseline models.

---

## Prediction Scenario
The trained model was applied to predict podium probabilities for the 2025 Chinese Grand Prix.
The results align with real-world performance trends, identifying dominant drivers and teams as top podium contenders.

---

## Project Structure
```text
PROJECT STRUCTURE
F1-Podium-Prediction/
|
|-- Scripts/
|   |-- Final.py
|
|-- Datasets/
|   |-- results.csv
|   |-- races.csv
|   |-- constructors.csv
|   |-- drivers.csv
|
|-- .gitignore
|-- README.md
 ```
---
## HOW TO RUN THE PROJECT
1. Clone the repository:
   git clone https://github.com/shubhamHULAGABALI/F1-Podium-Prediction.git

2. Navigate to the project directory:
   cd F1-Podium-Prediction

3. Install required dependencies:
   pip install pandas numpy scikit-learn matplotlib

4. Run the script:
   python Scripts/Final.py
---
## TECHNOLOGIES USED
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Git
- GitHub
---
## CONCLUSION
The Random Forest model demonstrates reliable performance in predicting
Formula 1 podium finishes using historical data. The results highlight
the importance of constructor strength, driver consistency, and
circuit-specific factors in determining race outcomes.

This project validates the effectiveness of ensemble learning techniques
in real-world sports analytics applications.

---
### FUTURE WORK
- Integration of weather and tire strategy data
- Evaluation of advanced ensemble models such as XGBoost or LightGBM
- Improved podium recall using resampling techniques
- Extension to real-time or live race predictions



