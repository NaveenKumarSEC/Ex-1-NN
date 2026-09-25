<H3>ENTER YOUR NAME: Naveenkumar M</H3>
<H3>ENTER YOUR REGISTER NO. 212224230182</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1: Importing the Libraries: Import the required Python libraries needed for data analysis and machine learning.<BR>

STEP 2: Importing the Dataset: Load the dataset into the program for processing and analysis.<BR>

STEP 3: Taking Care of Missing Data: Handle missing values by removing or replacing them with suitable values.<BR>

STEP 4: Encoding Categorical Data: Convert categorical (text) data into numerical format for machine learning models.<BR>

STEP 5: Normalizing the Data: Scale the features to a common range to improve model performance.<BR>

STEP 6: Splitting the Data into Training and Testing Sets: Divide the dataset into training and testing sets to train and evaluate the model.<BR>


##  PROGRAM:
```python
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv("/content/Churn_Modelling.csv")         # Read the dataset from drive
# Display first 5 rows
df.head()

# Finding Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Handling Missing Values
# Fill numerical columns with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Fill categorical columns with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# Check for Duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Detect Outliers (IQR Method)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} Outliers")

# Normalize the dataset
# Drop non-numeric/string columns before scaling
df_numeric = df.select_dtypes(include=['int64', 'float64'])

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric),
                         columns=df_numeric.columns)

print("\nNormalized Dataset:")
print(df_scaled.head())

# Split the dataset into input and output
X = df_scaled.drop('Exited', axis=1)
y = df_scaled['Exited']

# Splitting the data for Training & Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Print the training data and testing data
print("\nTraining Data (X_train):")
print(X_train)

print("\nTesting Data (X_test):")
print(X_test)

print("\nTraining Output (y_train):")
print(y_train)

print("\nTesting Output (y_test):")
print(y_test)

```

## OUTPUT:

## Finding Missing Values:

<img width="151" height="218" alt="image" src="https://github.com/user-attachments/assets/ce748d5d-21f9-434b-b5c5-7c6be9056d7a" />

## Handling Missing values:

<img width="208" height="227" alt="image" src="https://github.com/user-attachments/assets/072bf5ea-872e-4ce2-8278-e027a51bfddc" />

## Check for Duplicates & Outliers:

<img width="192" height="167" alt="image" src="https://github.com/user-attachments/assets/221d308d-607c-44df-91db-0cf43586fad6" />

## Normalize the dataset:

<img width="427" height="197" alt="image" src="https://github.com/user-attachments/assets/5e8a33a6-174b-416a-ac52-55d7af5db6d2" />

## splitting the data for training & Testing:

<img width="292" height="502" alt="image" src="https://github.com/user-attachments/assets/2d79d048-ece0-4678-86b9-75ae8c38cbf8" />

## Print the training data and testing data:

<img width="302" height="360" alt="image" src="https://github.com/user-attachments/assets/5e8cb534-5c45-40dd-a662-fa75e7393305" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


