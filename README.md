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

# MOdule 3 
Is there a relationship between categorical variables and the stroke ( which is the target)

PANDAS

-- we can use df['colum-name']..value_counts() --> It is used to show the count of different categorical data of that column --> It summarizes the categorical data
<img width="287" alt="image" src="https://github.com/user-attachments/assets/14067a30-8553-4d4f-b867-aa7b8060da46" />


BOX PLOTS

 Box plots are a great way to visualize numeric data, since you can visualize the various distributions of the data. The main features that the box plot shows are the median of the data which represents where the middle data point is, the upper quartile shows where the 75th percentile is, the lower quartile shows where the 25th percentile is. The data between the upper and lower quartile represents the inter-quartile range. Next, you have the lower and upper extremes. These are calculated as 1.5 times the inter-quartile range above the 75th percentile, and as 1.5 times the IQR below the 25th percentile. Finally, box plots also display outliers as individual dots that occur outside the upper and lower extremes. With box plots, you can easily spot outliers and also see the distribution and skewness of the data. Box plots make it easy to compare between groups. 
 <img width="365" alt="image" src="https://github.com/user-attachments/assets/57c57bb6-da37-4cc7-a563-a0cbb43341b2" />

 1. Symmetry:
If the box plot is symmetrical, the median will be roughly in the center of the box, and both whiskers will be about the same length. This indicates no skewness (i.e., a normal distribution).

2. Right (Positively) Skewed:
If the right whisker (the one extending to the larger values) is much longer than the left whisker, and the median is closer to the bottom (or left) of the box, the data is positively skewed (skewed to the right).

This means there are a few high values (outliers) pulling the distribution to the right.

3. Left (Negatively) Skewed:
If the left whisker (the one extending to the smaller values) is much longer than the right whisker, and the median is closer to the top (or right) of the box, the data is negatively skewed (skewed to the left).

This indicates that there are a few low values (outliers) pulling the distribution to the left.

4. Outliers:
Outliers (points beyond the whiskers) may further indicate skewness, particularly if they're concentrated in one direction (either high or low values).

Summary:
Symmetrical box plot: No skewness (normal distribution).

Right skew: Longer right whisker, median closer to the left.

Left skew: Longer left whisker, median closer to the right.

#BOX PLOTS
# Create box plots for numerical columns
sns.boxplot(data=df, x='stroke', y='age')
plt.title('Box plot of Age vs Stroke')      
plt.xlabel('Stroke')
plt.ylabel('Age')   
plt.show()

The output:
<img width="448" alt="image" src="https://github.com/user-attachments/assets/f964f72f-6fc7-483d-9041-1c075ca28395" />

-- We can use Panda dataframe.groupby() method to the categorical variables
-- groups data into subsets according to the different categories of that variable
-- We can group by a single variable or multiple variable, for multiple variables we have to pass multiple variable names


 



Play


00:00
00:00
Mute

Settings
Data Visualization commands in Python
cognitiveclass.ai
Estimated Effort: 20 mins
Visualizations play a key role in data analysis. In this reading, you'll be introduced to various forms of graphs and plots that you can create with your data in Python that help you in visualising your data for better analysis.

The two major libraries used to create plots are matplotlib and seaborn. We will learn the prominent plotting functions of both these libraries as applicable to Data Analysis.

Importing libraries
You can import the above mentioned libraries as shown below.

a. matplotlib

1
from matplotlib import pyplot as plt

Copied!
Alternatively, the command can also be written as:

1
import matplotlib.pyplot as plt

Copied!
Note that most of the plots that are of interest to us in this library are contained in the pyplot subfolder of the package.

matplotlib functions return a plot object which requires additional statements to display. While using matplotlib in Jupyter Notebooks, we require the graph to be displayed inside the notebook interface itself. It is, therefore, essential to add the following 'magic' statement after loading the library.

1
%matplotlib inline

Copied!
b. seaborn
Seaborn is usually imported in a code using the following statement:

1
import seaborn as sns

Copied!
matplotlib functions
1. Standard Line Plot

The simplest and most fundamental plot is a standard line plot. The function expects two arrays as input, x and y, both of the same size. x is treated as an independent variable and y as the dependent one. The graph is plotted as shortest line segments joining the x,y point pairs ordered in terms of the variable x.

Syntax:

1
plt.plot(x,y)

Copied!
A sample plot is shown in the image below.

Plot example.
2. Scatter plot

Scatter plots are graphs that present the relationship between two variables in a data set. It represents data points on a two-dimensional plane. The independent variable or attribute is plotted on the X-axis, while the dependent variable is plotted on the Y-axis.

Scatter plots are used in either of the following situations:

When we have paired numerical data
When there are multiple values of the dependent variable for a unique value of an independent variable
In determining the relationship between variables in some scenarios
Syntax:

1
plt.scatter(x,y)

Copied!
Here, x contains the independent variable, and y contains the dependent variable. You have the option to change the size, color, and shape of the markers with additional attributes in the function.
A sample scatter plot is shared below.
Sample scatter plot.

3. Histogram

A histogram is an important visual representation of data in categorical form. To view the data in a "Binned" form, we may use the histogram plot with a number of bins required or even with the data points that mark the bin edges. The x-axis represents the data bins, and the y-axis represents the number of elements in each of the bins.

Syntax:

1
plt.hist(x,bins)

Copied!
An example of a histogram plot is shown below. Use an additional argument, edgecolor, for better clarity of plot.
Consider the graph shown below. The left graph is the histogram plot for a data set, plotted without setting the edgecolor. The right one is the same graph but has the edgecolor argument set as the color black.

Histogram plot example.

4. Bar plot

A bar plot is used for visualizing catogorical data. The y-axis represents the average value of data points belonging to a particular category, while the x-axis represents the number of elements in the different categories.

Syntax:

1
plt.bar(x,height)

Copied!
Here, x is the categorical variable, and height is the number of values belonging to the category. You can adjust the width of each bin using an additional width argument in the function.

A sample graph is shown below.

Sample bar plot graph.

5. Pseudo Color Plot

A pseudocolor plot displays matrix data as an array of colored cells (known as faces). This plot is created as a flat surface in the x-y plane. The surface is defined by a grid of x and y coordinates that correspond to the corners (or vertices) of the faces. Matrix C specifies the colors at the vertices. The color of each face depends on the color of one of its four surrounding vertices. Of the four vertices, the one that comes first in the x-y grid determines the color of the face.

In this course, you use the pcolor plot for visualizing the contents of a pivot table that has been grouped on the basis of 2 parameters. Those parameters then represent the x and y-axis components that create the grid. The values in the pivot table are the average values of a third parameter. These values act as the code for the color the cell is going to take.

Syntax:

1
plt.pcolor(C)

Copied!
You can define an additional cmap argument to specify the color scheme of the plot.

Two sample pcolor plots are shown below, created for same data but for different color schemes.

Sample pcolor plots.

seaborn functions
1. Regression plot

A regression plot draws a scatter plot of two variables, x and y, and then fits the regression model and plots the resulting regression line along with a 95% confidence interval for that regression. The x and y parameters can be shared as the dataframe headers to be used, and the data frame itself is passed to the function as well.

Syntax:

1
sns.regplot(x = 'header_1',y = 'header_2',data= df)

Copied!
A sample regression plot is shared below.

Sample regression plot.

2. Box and whisker plot

A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables or across levels of a categorical variable. The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, except for points that are determined to be "outliers".

Consider the Box and whisker plot interpretation figure shown below.

Box and whisker plot example.
The plot uses whiskers to represent Minimum value to 25% quartile data and 75% quartile to Maximum value data. The range between 25% quartile and 75% quartile is considered as the Inter-Quartile Range. Outliers are generally classified as being outside 1.5 times the interquartile range.

A sample box plot is shown below
Box plot example.

3. Residual Plot

A residual plot is used to display the quality of polynomial regression. This function will regress y on x as a polynomial regression and then draw a scatterplot of the residuals.
Residuals are the differences between the observed values of the dependent variable and the predicted values obtained from the regression model. In other words, a residual is a measure of how much a regression line vertically misses a data point, meaning how far off the predictions are from the actual data points.

Syntax:

1
sns.residplot(data=df,x='header_1', y='header_2')

Copied!
Alternatively:

1
sns.residplot(x=df['header_1'], y=df['header_2'])

Copied!
A sample plot is shown below.

Residual plot example.

4. KDE plot

A Kernel Density Estimate (KDE) plot is a graph that creates a probability distribution curve for the data based upon its likelihood of occurrence on a specific value. This is created for a single vector of information. It is used in the course in order to compare the likely curves of the actual data with that of the predicted data.

Syntax:

1
sns.kdeplot(X)

Copied!
A sample graph made for a random set of values is shown below.

KDE plot examplke.

5. Distribution Plot

This plot has the capacity to combine the histogram and the KDE plots. This plot creates the distribution curve using the bins of the histogram as a reference for estimation. You can optionally keep or discard the histogram from being displayed. In the context of the course, this plot can be used interchangeably with the KDE plot.

Syntax:

1
sns.distplot(X,hist=False)

Copied!
Here, keeping the argument hist as True would plot the histogram along with the distribution plot. Both variations are shown in the image below.

Histogram and KDE sample plots.

Conclusion
This concludes the summary of the different types of plots being used in this course for the purpose of visualization.


