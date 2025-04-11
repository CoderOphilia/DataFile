<img width="483" alt="image" src="https://github.com/user-attachments/assets/4b86c80d-c32d-4941-9944-f5a5557c9204" />
# Data pre-processing
It is the process of converting or mapping data from the initial 'raw' form into another format in order to prepare the data for further analysis. The other names for data pre-processing are data cleaning and data wrangling.

Missing values occu when no data value is stored for a variable(feature) in an observation. Could be represnted as "?", "N/A", or 0 or just a blank cell.

# How to deal with missing data?
1. Check with the data collection source
2. Drop the missing data
   - Drop the variable
   - Drop the data entry: This is the best option if there is no much of data loss where it has least amount of impact.
3. Replace the missing values: This is the better option but the accuracy might be compromised. The value can be replace with an average or frequency or replace it based on the other function
4. Leave it as missing data

In python, we have a method called as dataframe.dropna() to drop the missing values. Specify axis = 0 to drop the entire row and axis = 1 to drop the entire column.
example:  df.dropna(subset=["bmi"], axis = 0, inplace = True)
when we set the inplace = True; we are allowing the modification to be done on the data set directly.

We can replace the missing value by using the datframe.replace(missing_value, new_value)
1. For replacing, first we have to normalize the data. mean = df["normalized-loses"].mean()
2. df["normalized-loses"].replace(np.nan.mean)
.replace(A, B, inplace = True) 
to replace A by B.


# Data Formatting
- Data is usually colected from different places and stored in different formats
- Bringing data into a common standard of expression allows users to make meaningful comparison

# Correcting Data types
- use dataframe.dtypes() to identify data type
- To convert data types: use dataframe.astype() to convert data type
- Example convert data type to integer in column "price"
- df["price"] = df["price"].astype("int")

# Data Normalization
Normalization enables a fairer comparison between the different features. It make sure that the features have the same impact and it is also important for computtational reasons.

- It helps to make sure that the range of the data is consistent
  
## Methods of normalizating data
1. Simple feature scaling: This method makes the value range betwwen 0 and 1
2. Min-Max method
3. Z-score or standard score
<img width="373" alt="image" src="https://github.com/user-attachments/assets/689cbf92-a9c6-4b00-aa09-b0cdb3372874" />
<img width="368" alt="image" src="https://github.com/user-attachments/assets/e21706ab-f242-4065-b291-56e9d0983764" />
<img width="394" alt="image" src="https://github.com/user-attachments/assets/66bb58b2-b005-4077-850a-7837dd1d44de" />
<img width="402" alt="image" src="https://github.com/user-attachments/assets/bc05a8af-b46c-4364-a244-ed4e50974fc7" />

# Binning
- Binning: Grouping of values into 'bins'
- Converts numeric data into categorical variables
- Group a set of numeric values into a set of "bin"
- A smaller number of bins will have better understanding of the data distribution
<img width="379" alt="image" src="https://github.com/user-attachments/assets/d2fa7f15-5d33-4a96-b816-a8bebaa85da1" />
In this example; 4 means, 4 equal interval
<img width="356" alt="image" src="https://github.com/user-attachments/assets/e87d6c0d-5b33-4976-8940-0c5679c49266" />

- We have to use the python function called as linspace to allocate the bins
- we have to mention the range and how many bins we need
- bins = np.linspace(min(df['price"]), max(df["price"]),4)
- group_names = ["Low", "Medium", "High"] --> This is the bin names
- df["priced-binned"] = pd.cut(df["price"], bin, labels = group_names, include_lowest = True)
- After that visualize the data using histograms
  
# Categorical variables
- Most statistical models cannot take in the objects/strings as input
- We convert categorical data into numeric data.
  -> Add a dummy variable for each unique category
  -> Assign 0 or 1 in each category
**One-hot encoding** is a method used in machine learning and data preprocessing to convert categorical data (like colors, countries, or product types) into a numerical format that a machine learning model can understand.
One-hot means only one column will have 1 (hot) and the rest will be 0 (cold) for each row
- In python, we can use pandas.get_dummies() method. It converts categorical varibales to dummy variables (0 or 1)
import pandas as pd

# Sample data
df = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Blue']
})

# One-hot encoding
encoded_df = pd.get_dummies(df, columns=['Color'])

print(encoded_df)

# Summary

Data formatting is critical for making data from various sources consistent and comparable.

Master the techniques in Python to convert units of measurement, like transforming "city miles per gallon" to "city-liters per 100 kilometers" for ease of comparison and analysis.

Acquire skills to identify and correct data types in Python, ensuring the data is accurately represented for subsequent statistical analyses.

Data normalization helps make variables comparable and helps eliminate inherent biases in statistical models.

You can apply Feature Scaling, Min-Max, and Z-Score to normalize data and apply each technique in Python using pandas’ methods.

Binning is a method of data pre-processing to improve model accuracy and data visualization.

Run binning techniques in Python using numpy's "linspace" and pandas' "cut" methods, particularly for numerical variables like "price."

Utilize histograms to visualize the distribution of binned data and gain insights into feature distributions.

Statistical models generally require numerical inputs, making it necessary to convert categorical variables like "fuel type" into numerical formats.

You can implement the one-hot encoding technique in Python using pandas’ get_dummies method to transform categorical variables into a format suitable for machine learning models.


## 🔧 What is `ColumnTransformer`?

`ColumnTransformer` lets you apply **different preprocessing steps to different columns** at the same time — like scaling numerical features and encoding categorical ones — all in one go.

---

### 👇 Code Breakdown:
```python
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])
```

This means:

### 1. `'num'`: For **numerical columns**
```python
StandardScaler()  # Standardizes the data (mean=0, std=1)
```
- Makes features like `age`, `bmi`, and `glucose` scale equally.
- Important for models like Logistic Regression and SVM.

### 2. `'cat'`: For **categorical columns**
```python
OneHotEncoder(drop='first', handle_unknown='ignore')
```
- Converts categories like `Male/Female`, `Urban/Rural` into binary 0/1 columns.
- `drop='first'` avoids dummy variable trap (drops the first category).
- `handle_unknown='ignore'` avoids errors if test data has unseen categories.

---

### ✅ What Does It Do Overall?

After running:
```python
X_transformed = preprocessor.fit_transform(X_model)
```

You get:
- All categorical variables turned into 0/1 columns
- All numerical variables scaled properly
- A **numerical-only, model-ready dataset**

---

### 🧠 Why It’s Useful:

| Without ColumnTransformer | With ColumnTransformer       |
|---------------------------|------------------------------|
| Manual encoding/scaling   | Everything handled together ✅ |
| Hard to scale/track       | Easy to manage and reuse     |
| Risk of errors             | Clean and consistent         |

---

