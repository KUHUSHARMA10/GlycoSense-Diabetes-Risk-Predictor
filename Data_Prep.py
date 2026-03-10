import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
#print("Libraries installed successfully!")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.utils import resample

#df= pd.read_csv("diabetes_prediction_dataset.csv")
#EDA

"""
print(df.head())
print(df.tail())
print(df.sample(5))
print(df.info())
print(df.describe())
print(df.shape) which is (100000, 9)
print(df.isnull().sum())
print(df.duplicated().sum()) # around 3854
print( df.drop_duplicates() ) [96146 rows x 9 columns]
print(df.isnull().sum())

print(df.mean(numeric_only=True))
 age                     41.885856
hypertension             0.074850
heart_disease            0.039420
bmi                     27.320767
HbA1c_level              5.527507
blood_glucose_level    138.058060
diabetes                 0.085000
dtype: float64

print(df.std(numeric_only=True))
age                    22.516840
hypertension            0.263150
heart_disease           0.194593
bmi                     6.636783
HbA1c_level             1.070672
blood_glucose_level    40.708136
diabetes                0.278883
dtype: float64

print(df.min(numeric_only=True))
age                     0.08
hypertension            0.00
heart_disease           0.00
bmi                    10.01
HbA1c_level             3.50
blood_glucose_level    80.00
diabetes                0.00
dtype: float64

print(df.describe())
  age  hypertension  heart_disease            bmi    HbA1c_level  blood_glucose_level       diabetes
count  100000.000000  100000.00000  100000.000000  100000.000000  100000.000000        100000.000000  100000.000000
mean       41.885856       0.07485       0.039420      27.320767       5.527507           138.058060       0.085000
std        22.516840       0.26315       0.194593       6.636783       1.070672            40.708136       0.278883
min         0.080000       0.00000       0.000000      10.010000       3.500000            80.000000       0.000000
25%        24.000000       0.00000       0.000000      23.630000       4.800000           100.000000       0.000000
50%        43.000000       0.00000       0.000000      27.320000       5.800000           140.000000       0.000000
75%        60.000000       0.00000       0.000000      29.580000       6.200000           159.000000       0.000000
max        80.000000       1.00000       1.000000      95.690000       9.000000           300.000000       1.000000
"""
#HANDLE MISSING VALUES 
#df = df.fillna(df.mean)
#print (df.isna().sum()) all 0
#print(df.corr(numeric_only=True))

#SAVE CLEANED DATASET
#df.to_csv("cleaned_diabetes.csv", index=False)

df = pd.read_csv("cleaned_diabetes.csv")

# ===============================
# GROUPBY ANALYSIS (EDA)
# ===============================

print(df.groupby("diabetes").size())

print(df.groupby("diabetes")["blood_glucose_level"].mean())

print(df.groupby("diabetes")["HbA1c_level"].mean())

print(df.groupby("gender")["diabetes"].mean())

print(df.groupby("smoking_history")["diabetes"].mean())

print(df.groupby("hypertension")["diabetes"].mean())

print(df.groupby("heart_disease")["diabetes"].mean())


# ===============================
# FEATURE ENGINEERING
# ===============================

# Diabetes thresholds
df["high_hba1c"] = (df["HbA1c_level"] >= 6.5).astype(int)
df["high_glucose"] = (df["blood_glucose_level"] >= 200).astype(int)

# Prediabetes ranges
df["prediabetes_hba1c"] = ((df["HbA1c_level"] >= 5.7) & (df["HbA1c_level"] < 6.5)).astype(int)

df["prediabetes_glucose"] = ((df["blood_glucose_level"] >= 126) & (df["blood_glucose_level"] < 200)).astype(int)

# Obesity indicator
df["obese"] = (df["bmi"] >= 30).astype(int)


# BALANCE DATASET
df_majority = df[df.diabetes == 0]
df_minority = df[df.diabetes == 1]

# Upsample minority class
df_minority_upsampled = resample(
    df_minority,
    replace=True,             # allow duplicates
    n_samples=len(df_majority), # match majority size
    random_state=42
)

# Combine
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Shuffle dataset
df = df_balanced.sample(frac=1, random_state=42)

# Features
X = df.drop("diabetes", axis=1)

# Target
y = df["diabetes"]
#X = all columns except diabetes
#y = diabetes

X = pd.get_dummies(X, drop_first=True)


#corr = df.corr(numeric_only=True)
#print(corr["diabetes"].sort_values(ascending=False))
"""
output
diabetes               1.000000
blood_glucose_level    0.419558
HbA1c_level            0.400660
age                    0.258008
bmi                    0.214357
hypertension           0.197823
heart_disease          0.171727
Name: diabetes, dtype: float64

So the most important features are:
blood_glucose_level
HbA1c_level
age
bmi
"""

"""
VISUALIZE CORRELATION
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(numeric_only=True), annot=True)

plt.title("Correlation Heatmap", fontsize=20, family="Arial", fontweight="bold", color="#000000")

plt.show()
"""
#MULTICOLINERARITY 
#corr_matrix = X.corr(numeric_only=True)
#print(corr_matrix)
"""
output of that 
diabetes               1.000000
blood_glucose_level    0.419558
HbA1c_level            0.400660
age                    0.258008
bmi                    0.214357
hypertension           0.197823
heart_disease          0.171727
Name: diabetes, dtype: float64
                          age  hypertension  heart_disease       bmi  HbA1c_level  blood_glucose_level
age                  1.000000      0.251171       0.233354  0.337396     0.101354             0.110672
hypertension         0.251171      1.000000       0.121262  0.147666     0.080939             0.084429
heart_disease        0.233354      0.121262       1.000000  0.061198     0.067589             0.070066
bmi                  0.337396      0.147666       0.061198  1.000000     0.082997             0.091261
HbA1c_level          0.101354      0.080939       0.067589  0.082997     1.000000             0.166733
blood_glucose_level  0.110672      0.084429       0.070066  0.091261     0.166733             1.000000

features are not strongly correlated with each other
"""
#FEATURE SCALING

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# save scaler
pickle.dump(scaler, open("scaler.pkl","wb"))

#TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
"""
80% → training data
20% → testing data
"""