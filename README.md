# Heart Disease Detection

This project is a machine learning model that predicts the likelihood of heart disease using logistic regression. The dataset is sourced from the UCI Machine Learning Repository, and the model evaluates its performance based on accuracy, a classification report, and a confusion matrix.

## Dataset

The dataset used for this project contains various medical information about patients, which is used to predict the presence of heart disease. You can download the dataset from the UCI repository:  
[Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

### Features

The dataset includes the following features:

- **age**: Age in years
- **sex**: Gender (1 = male; 0 = female)
- **cp**: Chest pain type (4 types)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar (> 120 mg/dl) (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)

## Requirements

Install the necessary libraries using `pip`:

```bash
pip install numpy pandas scikit-learn
The project uses the following key libraries:

pandas: For data manipulation
numpy: For numerical computations
scikit-learn: For machine learning modeling and evaluation

How to Run
Clone the repository.
Navigate to the project folder.
Ensure that the dataset (heart_disease.csv) is placed in the project directory.
Run the Python script:
python Heart Disease Detector!.py

Model Output
Once the script runs successfully, it evaluates the model's performance and outputs:

Accuracy of the Logistic Regression Model:
Displays the accuracy of the predictions in percentage form.

Classification Report:
Detailed report that includes precision, recall, and F1-score for each class (heart disease presence/absence).

Confusion Matrix:
A matrix showing the model's correct and incorrect classifications.

Sample Output
Accuracy of the Logistic Regression model: 80.67%

Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.79      0.82        40
           1       0.75      0.81      0.78        31

    accuracy                           0.81        71
   macro avg       0.80      0.80      0.80        71
weighted avg       0.81      0.81      0.81        71

Confusion Matrix:
[[31  9]
 [ 6 25]]

File Structure
.
├── heart+disease/
│   ├── dataset.csv                 # Original heart disease dataset
│   └── heart_disease.csv            # Cleaned and processed dataset
├── costs/                           # (Optional) Folder for cost-related metrics if needed
├── Heart Disease Detector!.py       # Main Python script
└── README.md                        # Project documentation

Future Work
Implement more advanced models (e.g., Decision Trees, Random Forests) for comparison.
Perform feature selection to reduce dimensionality.
Visualize important features using graphs for better interpretability.

Acknowledgments
Dataset from UCI Machine Learning Repository: Heart Disease Dataset

License
This project is licensed under the MIT License.

This `README.md` outlines the purpose of the project, how to run it, what output to expect, and the structure of the project files. You can update any specific details if necessary.
