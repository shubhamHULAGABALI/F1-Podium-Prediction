import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load datasets
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print("BASE_DIR:", BASE_DIR)

    DATASETS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Datasets'))
    print("DATASETS_DIR:", DATASETS_DIR)

    results = pd.read_csv(os.path.join(DATASETS_DIR, 'results.csv'))
    races = pd.read_csv(os.path.join(DATASETS_DIR, 'races.csv'))
    constructors = pd.read_csv(os.path.join(DATASETS_DIR, 'constructors.csv'))
    drivers = pd.read_csv(os.path.join(DATASETS_DIR, 'drivers.csv'))

    print("Datasets loaded successfully.")

except FileNotFoundError as e:
    print("File loading error:")
    print(e)
    exit()

# Merge datasets to include race, constructor, and driver details
data = results.merge(races[['raceId', 'year', 'circuitId']], on='raceId', how='left')
data = data.merge(constructors[['constructorId', 'name']], on='constructorId', how='left')
data = data.merge(drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')

# Filter for recent years (2022–2024) for modern F1 relevance
data = data[data['year'].isin([2022, 2023, 2024])]

# Define target: 1 if driver finished in top 3 (podium), 0 otherwise
data['podium'] = (data['positionOrder'] <= 3).astype(int)

# Feature engineering: Aggregate team and driver statistics per year
team_stats = data.groupby(['year', 'constructorId']).agg({
    'grid': 'mean',          # Average starting grid position
    'points': 'sum',         # Total constructor points
    'positionOrder': 'mean'  # Average finishing position
}).reset_index()
team_stats.columns = ['year', 'constructorId', 'avg_grid', 'total_points', 'avg_position']

# Driver statistics: Aggregate per driver per year
driver_stats = data.groupby(['year', 'driverId']).agg({
    'points': 'sum',         # Total driver points
    'podium': 'sum',         # Number of podiums
    'raceId': 'count'        # Number of races participated
}).reset_index()
driver_stats.columns = ['year', 'driverId', 'driver_points', 'past_podiums', 'races_participated']

# Merge team and driver stats back to main dataset
data = data.merge(team_stats, on=['year', 'constructorId'], how='left')
data = data.merge(driver_stats, on=['year', 'driverId'], how='left')

# Select features and target
features = ['avg_grid', 'total_points', 'avg_position', 'circuitId', 'driver_points', 'past_podiums', 'races_participated']
X = data[features].copy()  # Create a copy to avoid SettingWithCopyWarning
y = data['podium']

# Encode categorical feature (circuitId)
le_circuit = LabelEncoder()
X['circuitId'] = le_circuit.fit_transform(X['circuitId'])

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# Evaluate model with comprehensive metrics
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Cross-validation score for robustness
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy (5-fold): {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

# Classification report for precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Podium', 'Podium']))

# Confusion matrix for detailed performance analysis
print("\nConfusion Matrix (TN, FP, FN, TP):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Predict podium probabilities for 2025 Chinese Grand Prix (circuitId=17, Shanghai)
next_race = team_stats[team_stats['year'] == 2024][['constructorId', 'avg_grid', 'total_points', 'avg_position']].copy()
next_race_driver = driver_stats[driver_stats['year'] == 2024][['driverId', 'driver_points', 'past_podiums', 'races_participated']].copy()
next_race = next_race.merge(data[['constructorId', 'driverId']].drop_duplicates(), on='constructorId', how='left')
next_race = next_race.merge(next_race_driver, on='driverId', how='left')
next_race['circuitId'] = 17  # Shanghai International Circuit
next_race['circuitId'] = le_circuit.transform([17])[0]

# Predict podium probabilities
probs = rf.predict_proba(next_race[features])[:, 1]
next_race['podium_prob'] = probs

# Map constructor and driver names
next_race['constructor_name'] = next_race['constructorId'].map(constructors.set_index('constructorId')['name'])
next_race['driver_name'] = next_race['driverId'].map(drivers.set_index('driverId')['forename'] + ' ' + drivers.set_index('driverId')['surname'])

# Sort by podium probability and format output
next_race_sorted = next_race[['constructor_name', 'driver_name', 'podium_prob']].sort_values(by='podium_prob', ascending=False)
next_race_sorted['podium_prob'] = (next_race_sorted['podium_prob'] * 100).round(2)  # Convert to percentage

# Display podium probabilities for all teams and drivers
print("\nPodium Probabilities for 2025 Chinese Grand Prix (All Teams and Drivers):")
print("-" * 60)
for index, row in next_race_sorted.iterrows():
    print(f"{row['constructor_name']:<20} | {row['driver_name']:<20} | {row['podium_prob']:>6.2f}%")
print("-" * 60)

# Identify predicted top podium contender
top_contender = next_race.loc[next_race['podium_prob'].idxmax()]
team = top_contender['constructor_name']
driver = top_contender['driver_name']
print(f"\nPredicted Top Podium Contender: {driver} ({team}) (Probability: {top_contender['podium_prob']:.2f}%)")