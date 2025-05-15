
# 📦 Data analysis and visualization libraries
import pandas as pd                            # For data loading and manipulation
import numpy as np                             # For numerical computations


# 🧹 Data preprocessing and cleaning
from datasist.structdata import detect_outliers        # To detect outliers in data
from sklearn.impute import SimpleImputer              # For missing value imputation (simple methods)
from sklearn.impute import KNNImputer                 # For missing value imputation (KNN-based)
from sklearn.preprocessing import RobustScaler        # For scaling features robust to outliers
from sklearn.preprocessing import OneHotEncoder       # For encoding categorical features
from sklearn.compose import ColumnTransformer         # For applying transformers to specific columns

# ⚖️ Handling imbalanced data
from imblearn.over_sampling import SMOTE              # For oversampling minority class
from imblearn.under_sampling import RandomUnderSampler  # For undersampling majority class
from imblearn.pipeline import Pipeline                # Pipeline supporting sampling steps

# 🧪 Data splitting and cross-validation
from sklearn.model_selection import train_test_split      # For splitting data into train and test sets
from sklearn.model_selection import StratifiedKFold       # For stratified K-Fold cross-validation
from sklearn.model_selection import cross_val_score, cross_val_predict  # For cross-validation

# 🧮 Classification models
from sklearn.linear_model import LogisticRegression       # Logistic Regression model
from sklearn.svm import SVC                               # Support Vector Machine model
from sklearn.neighbors import KNeighborsClassifier        # K-Nearest Neighbors model
from sklearn.tree import DecisionTreeClassifier           # Decision Tree model
from sklearn.ensemble import RandomForestClassifier       # Random Forest model
from sklearn.ensemble import GradientBoostingClassifier   # Gradient Boosting model
from sklearn.ensemble import HistGradientBoostingClassifier  # Fast version of Gradient Boosting
from xgboost import XGBClassifier                         # XGBoost model
from catboost import CatBoostClassifier                   # CatBoost model

# 🔍 Model evaluation and tuning
from sklearn.model_selection import GridSearchCV          # For hyperparameter tuning
from sklearn.metrics import accuracy_score                # For calculating accuracy
from sklearn.metrics import classification_report         # For detailed classification performance
from sklearn.metrics import confusion_matrix              # For generating confusion matrix

# 💾 Saving and loading models
import joblib  



df = pd.read_csv("diabetes_prediction_dataset.csv")



# It removes all rows where the BMI is less than or equal to  or greater than or equal  70.
# These values are considered unrealistic or outliers

df.drop(index= df[(df["bmi"] <= 10) | (df["bmi"] >= 70)].index , inplace=True )
df.reset_index(drop=True , inplace=True)



# set the value to "never" for all rows where the person's age is 15 years old or younger,
# assuming that children at that age do not smoke.
df.loc[df["age"] <= 15 , "smoking_history" ] = "never"  


# "is_fat" 
# If the BMI is 25 or higher, the function returns 1 (meaning the person is considered fat).
# Otherwise, it returns 0 (meaning the person is not fat).

def is_the_person_fat(bmi):
    
    if bmi >= 25 :
        return 1 
    else :
        return 0
df["is_fat"] = df["bmi"].apply(is_the_person_fat)



# df["smoking_history"].unique() = ['never', 'No Info', 'current', 'former', 'ever', 'not current']

smoking_risk_map = {
    "never": 0,
    "former": 1,
    "not current": 2,
    "ever": 3,
    "current": 4,
    "No Info": np.nan # I will replace No Info
}
df["smoking_risk_level"] = df["smoking_history"].map(smoking_risk_map)



imputer = KNNImputer(n_neighbors=5) # imputer for fill no info data 

df_for_imputation = df.drop(columns=["gender" , "smoking_history"]) # Prepare dataframe without categorical column



imputed_array = imputer.fit_transform(df_for_imputation) # fill the nan values in the data

df_imputed = pd.DataFrame(imputed_array, columns=df_for_imputation.columns) # make the rezalt as df

df = pd.concat([df[["gender", "smoking_history"]].reset_index(drop=True),
                      df_imputed.reset_index(drop=True)], axis=1) # concat the categorical column with the df

df["smoking_risk_level"] = df["smoking_risk_level"].astype("int") # cange the type to be int 



reverse_map = {
    0: "never",
    1: "former",
    2: "not current",
    3: "ever",
    4: "current"
}
df["smoking_history"] = df["smoking_risk_level"].map(reverse_map) # Apply the result on the column

# (MS) Metabolic Syndrome - (IFG) Impaired Fasting Glucose - (DM) Diabetes Mellitus

# risk_age
# Older people are more susceptible to this disease.
# If the age is 60 or older, it returns 1 .
# Otherwise, it returns 0.



def is_high_risk_age (age):
    if age >= 60 :
        return 1
    else :
        return 0


df["risk_age"] = df["age"].apply(is_high_risk_age)


# glucose_risk_level
# that categorizes blood glucose levels into three risk levels:
# 0 → Normal blood sugar (≤ 140)
# 1 → Prediabetes (between 141 and 199)
# 2 → Diabetes (≥ 200 )

def is_high_risk_glucose(blood_glucose_level):
    if blood_glucose_level <= 140:
        return 0
    elif blood_glucose_level <= 199:
        return 1
    else :
        return 2

df["glucose_risk_level"] = df["blood_glucose_level"].apply(is_high_risk_glucose)


# that categorizes a person's HbA1c level into 4 risk levels:
# 0 → Normal (HbA1c < 5.7%)
# 1 → Slightly Elevated (5.7 ≤ HbA1c < 6%)
# 2 → Prediabetes (6% ≤ HbA1c < 6.5%)
# 3 → Diabetes (HbA1c ≥ 6.5%)

def hba1c_risk(val):
    if val < 5.7:
        return 0
    elif val < 6:
        return 1
    elif val < 6.5:
        return 2
    else:
        return 3

df["hba1c_risk_score"] = df["HbA1c_level"].apply(hba1c_risk)


# that calculates a composite risk score for each person by adding values from several health related columns .


def risk_score(row):
    score = (
        row["hypertension"]
        + row["heart_disease"]
        + row["is_fat"]
        + row["smoking_risk_level"]
        + row["risk_age"]
        + row["glucose_risk_level"]
        + row["hba1c_risk_score"]
    )
    return score

df["risk_score"] = df.apply(risk_score, axis=1)


# Change the type of binary column from float to integer
df[["hypertension" ,"heart_disease",
    "diabetes","is_fat","smoking_risk_level"
    ,"risk_age","glucose_risk_level",
    "hba1c_risk_score","risk_score"]] = df[["hypertension" ,"heart_disease","diabetes","is_fat",
                                            "smoking_risk_level","risk_age","glucose_risk_level",
                                            "hba1c_risk_score","risk_score"]].astype("int")



# a- Detect & Handle Duplicates
# remove duplicate values
df.drop_duplicates(inplace=True , ignore_index=True)




df.drop(index=detect_outliers(df , 0 ,["bmi","HbA1c_level","blood_glucose_level"]) , inplace=True)



# b- train_test_split
X = df.drop(columns=["diabetes"])
Y = df["diabetes"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.2, shuffle=True, stratify=Y, random_state=42)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape



# e- Encoding: (Ordinal:[OrdinalEncoder, LabelEncoder] - Nominal: [< 7 uniques(OneHotEncoding), > 7 uniques (BinaryEncoder)])

#column 1- gender i have ["Male" , "Female" , "Other"] and the those values are Nominal so i will ues OneHotEncoding 
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore' , drop=None))
])

#column 2- smoking_history i made smoking_risk_level column and it has the same values so if i make Encoding to 
# smoking_history the model will see the values more than one time and the model maby will overfit so i will drop it 
X_train.drop(columns=["smoking_history"] , inplace=True)
X_test.drop(columns=["smoking_history"] , inplace=True)



# g- Scaling: StandardScaler, MinMaxScaler, RobustScaler: X_train_resampled_scaled
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', RobustScaler())
])


# Making transformer to concatenate numerical pipeline and categorical pipeline
numeric_features = ["age", "bmi", "HbA1c_level", "blood_glucose_level", "risk_score"]
categorical_features = ['gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)


# i will ues Clustering model for extract future "The distance between each point And it's centre or class"

from sklearn.cluster import KMeans


X_combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y_combined = np.concatenate((Y_train, Y_test), axis=0)


KMeans_model = Pipeline([
        ("preprocessor" , preprocessor) ,
        ( 'smote' , SMOTE(random_state=42)),
        ('KM', KMeans(n_clusters=2 , max_iter=300 , n_init=20))
    ])

scor = 0 
while scor < 0.70 :


    KMeans_model.fit(X_combined, y_combined)

    cluster_labels = KMeans_model.predict(X_combined)
    scor = accuracy_score(y_combined, cluster_labels)



X_train_new_future = pd.DataFrame(KMeans_model.transform(X_train), columns=["distance_to_centroid_0", "distance_to_centroid_1"])
X_test_new_future = pd.DataFrame(KMeans_model.transform(X_test), columns=["distance_to_centroid_0", "distance_to_centroid_1"])

X_train = pd.concat([X_train.reset_index(drop=True), X_train_new_future.reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_new_future.reset_index(drop=True)], axis=1)


X_train["age_hba1c_interaction"] = X_train["age"] * X_train["HbA1c_level"]
X_train["bmi_glucose_interaction"] =  X_train["bmi"] * X_train["blood_glucose_level"]
X_train["bmi_age_ratio"] = X_train["bmi"] / X_train["age"]
X_train['glucose_triple_score'] = (X_train['blood_glucose_level'] * X_train['HbA1c_level']) / X_train['bmi']

X_test["age_hba1c_interaction"] = X_test["age"] * X_test["HbA1c_level"]
X_test["bmi_glucose_interaction"] =  X_test["bmi"] * X_test["blood_glucose_level"]
X_test["bmi_age_ratio"] = X_test["bmi"] / X_test["age"]
X_test['glucose_triple_score'] = (X_test['blood_glucose_level'] * X_test['HbA1c_level']) / X_test['bmi']


from sklearn.ensemble import StackingClassifier


numeric_features = ["age", "bmi", "HbA1c_level", "blood_glucose_level",
                    "risk_score" , "distance_to_centroid_0", "distance_to_centroid_1",
                    "age_hba1c_interaction","bmi_glucose_interaction","bmi_age_ratio" ,'glucose_triple_score']
categorical_features = ['gender']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore' , drop=None))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)
# ------------------------------------------------------------
final_catboost_pipeline= Pipeline([
        ("preprocessor" , preprocessor) ,
        ('smote', SMOTE(random_state=42)),
        ('CatBoost',  CatBoostClassifier(silent=True, depth = 4, iterations = 300, l2_leaf_reg =1, learning_rate = 0.2))
    ])
final_xgb_pipeline = Pipeline([
        ("preprocessor" , preprocessor) ,
        ( 'smote' , SMOTE(random_state=42)),
        ('XGB',XGBClassifier(eval_metric='logloss', random_state=42 , colsample_bytree = 1.0, learning_rate =0.2, max_depth = 3, n_estimators =300, subsample = 0.8))
    ])

final_HistGradient_pipeline = Pipeline([
        ("preprocessor" , preprocessor) ,
        ('smote', SMOTE(random_state=42)),
        ('HistGradient', HistGradientBoostingClassifier(random_state=42 , learning_rate=0.2,max_depth=10,max_iter=200))
    ])

#--------------------------------------------------------------------
hist_param_grid = {
    'learning_rate': [0.2],
    'max_iter': [200],
    'max_depth': [None,10]
}

final_hist_estimator = GridSearchCV(
    estimator=HistGradientBoostingClassifier(random_state=42),
    param_grid=hist_param_grid,
    cv=3,
    scoring='recall', # change to be recol 
    verbose=1,
    n_jobs=-1
)
#--------------------------------------------------------------------
Stacking_model = StackingClassifier(estimators= [('XGB' , final_xgb_pipeline),
                                ('HistGradient' , final_HistGradient_pipeline ),
                                ('CatBoost' , final_catboost_pipeline)]
                                ,final_estimator= final_hist_estimator, 
                                cv= 5, n_jobs=-1)
#--------------------------------------------------------------------
Stacking_model.fit(X_train , Y_train)


joblib.dump(Stacking_model, 'final_model.pkl')
