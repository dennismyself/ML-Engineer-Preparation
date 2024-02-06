# ML-Engineer-Preparation

## Python Knowledge 
* List Indexing: Doesn't include the right number
* Lambda
```
lambda arguments: expression 
```

* map(function, iterable)
	* iterates through the iterables, pass each iterable to the function one by one
```
numbers = [1, 2, 3, 4]
squared_numbers = map(lambda x: x ** 2, numbers)
print(list(squared_numbers))  # Output: [1, 4, 9, 16]
```

* filter(function, iterable)
```
numbers = [1, 2, 3, 4, 5, 6]
even_numbers = filter(lambda x: x % 2 == 0, numbers)
print(list(even_numbers))  # Output: [2, 4, 6]
```

* sorted(iterable, key=function)
```
pairs = [(1, 'one'), (3, 'three'), (2, 'two')]
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
print(sorted_pairs)  # Output: [(1, 'one'), (3, 'three'), (2, 'two')]
```
* ```for i, ele in enumerate(l):```

## Pandas
* All pandas methods: have the inplace attribute. False by default: True: Replace the original DF, False: 
* Create a new df
```
Df = pd.DataFrame([], columns = COLUMN NAME)
```

* Access and rename columns
```
df.columns = ['NewName1', 'NewName2']
df = df.rename(columns={'OldName1': 'NewName1', 'OldName2': 'NewName2'})
df.rename(columns={'OldName1': 'NewName1', 'OldName2': 'NewName2'}, inplace=True)
```

* Get the shape of a DF
```data.shape``` 
  * Returns a tuple. Shape is an attribute

* Display the first several data
  * head method: a method provided by the pandas library that is used on a DataFrame to return the first n rows. If n is omitted, it defaults to returning the first 5 rows. 

* Select Data
```
df.loc[row_labels, column_labels]

students.loc[students['student_id'] == 101, ['name', 'age']]
dataframe.iloc[row_indices, column_indices]
```

* Drop Duplicates
  * subset: This is the column label or sequence of labels to consider for identifying duplicate rows. If not provided, it considers all columns in the DataFrame.
  * keep: This argument determines which duplicate row to retain.
  * 'first': (default) Drop duplicates except for the first occurrence.
  * 'last': Drop duplicates except for the last occurrence.
  * False: Drop all duplicates.
  * inplace: If set to True, the changes are made directly to the object without returning a new object. If set to False (default), a new object with duplicates dropped will be returned.
```
newDF = customers.drop_duplicates(subset = "email", keep = "first", inplace = False)
```
		
* Drop Empty Cells
	*  dropna Function: The dropna function belongs to the pandas DataFrame and is used to remove missing values. Missing data in pandas is generally represented by the NaN
	*  dropna Function Argument Definition:
		*  axis: It can be {0 or 'index', 1 or 'columns'}. By default it's 0. If axis=0, it drops rows which contain missing values, and if axis=1, it drops columns which contain missing value.
		*  how: Determines if row or column is removed from DataFrame, when we have at least one NA or all NA.
			§ how='any' : If any NA values are present, drop that row or column (default).
			§ how='all' : If all values are NA, drop that row or column.
		*  thresh: Require that many non-NA values. This is an integer argument which requires a minimum number of non-NA values to keep the row/column.
		*  subset: Labels along the other axis to consider, e.g. if you are dropping rows these would be a list of columns to include. This is particularly useful when you only want to consider NA values in certain columns.
		*  inplace: It's a boolean which makes the changes in data frame itself if True. Always remember when using the inplace=True argument, you're modifying the original DataFrame. If you need to retain the original data for any reason, avoid using inplace=True and instead assign the result to a new DataFrame.
```
newdf = students.dropna(axis = 0, subset = "name")
```
* Change Data Type 
  	* astype Function: The astype function is used to cast a pandas object to a specified dtype (data type). astype can be used to cast a pandas object to any dtype. The astype function does not modify the original DataFrame in place. Instead, it returns a new DataFrame with the specified data type changes
  	* dtype: It's a data type, or dict of column name -> data type.
  	* copy: By default, astype always returns a newly allocated object. If copy is set to False, a new object will only be created if the old object cannot be casted to the required type.
  	* errors: Controls the raising of exceptions on invalid data for the provided dtype. By default, raise is set which means exceptions will be raised.

```
students = students.astype({'grade':int})
#OR
students["grade"] = students["grade"].astype("int")
```
* Fillna Function: fillna is a function in the pandas library, used primarily with pandas Series and DataFrame objects. It allows you to fill NA/NaN values using specified methods. In this context, we are using it to replace the None (or NaN in the usual dataframe representation) values.

```
products["quantity"].fillna(0, inplace = True)
```

* Concatenate
  * pd.concat(): A convenient function within pandas used to concatenate DataFrames either vertically (by rows) or horizontally (by columns).
		* The objs parameter is a sequence or mapping of Series or DataFrame objects to be concatenated.
		* The axis parameter determines the direction of concatenation:
			* axis=0 is set as the default value, which means it will concatenate DataFrames vertically (by rows).
			* axis=1 will concatenate DataFrames horizontally (by columns).
```
pd.concat([df1,df2], axis =0)
```
  
* The pivot Function
	*  index: It determines the rows in the new DataFrame. For this example, we use the month column from the original DataFrame as the index, which means our pivoted table will have one row for each unique value in the month column.
	* columns: It determines the columns in the new DataFrame. Here, we're using the city column, which means our pivoted table will have one column for each unique value in the city column.
	* values: This argument specifies the values to be used when the table is reshaped. For this example, we use the temperature column from the original DataFrame.
```
ans = weather.pivot(index='month', columns='city', values='temperature')
```
* Melt Function
	* melt Function: pandas' melt function is used to transform or reshape data. It changes the DataFrame from a wide format, where columns represent multiple variables, to a long format, where each row represents a unique variable.
	* After sorting the rows based on the weight, we're only interested in the name column for our final result. By using double square brackets [['name']], we select only this column. The double brackets ensure that the result is a DataFrame and not a Series.
	* Pandas "object" datatype; 
		a. String Data: By default, when pandas encounters a column that has text data (strings), it assigns the object data type to that column. This is because, under the hood, pandas stores strings in arrays of Python objects to accommodate the variable string length, as strings in Python are variable-length.
		b. Mixed Types: If a column has mixed types (numbers and strings, for example), pandas will default to using the object data type for safety, as it can store arbitrary Python objects.
Drop Columns 
```
 nyc_model = nyc_data.drop(columns=['name','id' ,'host_id','host_name', 'last_review','price'])
```

## Kaggle Airbnb Housing Price Prediction 
* Check how many null values: ```nyc_data.isnull().sum()```
* Seaborn
	* Seaborn is a Python visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.
```
import seaborn as sns
sns.scatterplot(x='room_type', y='price', data=nyc_data) #'room_type' and 'price' are column names 
```
* DataFrame as Input: Seaborn functions can take pandas DataFrame directly as input. You can specify the column names for x, y, hue, etc., and Seaborn automatically uses the DataFrame's data.
	
* Convert Columns Data from for feature engineering
```
nyc_data['neighbourhood_group'] = nyc_data['neighbourhood_group'].astype("category").cat.codes
```
	* Convert to categorical data first
	* Then convert the categories to numbers
	* ```.cat.codes``` is a simple and efficient way to transform categorical data into a numerical format, making it easier to use in various data analysis and machine learning contexts.

* Residual Plot: 
	* Does linear regression
	* Plot the difference between real and predicted value 
	* If the data evenly scattered along the x axis -> linear model is appropriate


* Principal Component Analysis (PCA)
  	* Variance Explanation: In the context of a correlation matrix, the eigenvalues indicate the amount of variance explained by each of the principal components (if you're performing PCA). A higher eigenvalue corresponds to a higher amount of variance explained by the principal component associated with that eigenvalue.
  	* Geometrically speaking, principal components represent the directions of the data that explain a maximal amount of variance, that is to say, the lines that capture most information of the data.
	* Application 1: Used to detect Multicollinearity
		* Multicollinearity is a statistical concept where several independent variables in a model are correlated. Two variables are considered perfectly collinear if their correlation coefficient is +/- 1.0. Multicollinearity among independent variables will result in less reliable statistical inferences.
		* Find eigen value and eign vector: multicollinearity
		```
		V=np.linalg.eig(corr)
  		# If none one of the eigenvalues of the correlation matrix is close to zero. It means that there is no multicollinearity exists in the data.
		```
 	* Application 2: Dimension Reduction
```
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Standarise the data to mean = 1, variance = 1
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
# Three main components
pca = PCA(n_components=3)
pca.fit(df_scaled)
# Transform the data to the new PCA space
df_pca = pca.transform(df_scaled)
```
* Split training and testing sets
```
X_train, X_test, y_train, y_test = train_test_split(nyc_model_x, nyc_model_y, test_size=0.3,random_state=42)
# random_state=42: Ensures reproducibility of the results. By setting a random state, you get the same split each time you run the code. 
```
* Extra Tree classifier
	* Constructing Multiple Trees. Ensemble learning 
   	* Different from Random Forest: For each tree, the entire dataset is used (unlike Random Forest which uses bootstrapped datasets).
   	* Each bootstrapped dataset is a random sample of the original dataset and is of the same size as the original. Due to the nature of sampling with replacement, some observations may be repeated in each bootstrapped dataset, and some observations from the original dataset may be left out.
   	* When constructing the trees, splits are chosen completely at random from the range of values in the sample at each split (unlike Random Forest which chooses the optimal split among a random subset of features).
   	* The final prediction is made by averaging the predictions of the individual decision trees (for regression) or by taking the majority vote (for classification).
```
transofrmed_y = preprocessing.LabelEncoder().fit_transform(y_train)
feature_model = ExtraTreesClassifier(n_estimators=50)
feature_model.fit(X_train,transofrmed_y)
```
* Grid search
```
from sklearn.liner_model import linear_regression
from sklearn.model_selection import GridSearchCV
model = linear_regression()
para = {}
grid_search_LR = GridSearchCV(estimator = model, para_grod = para, cv = cv)
grid_search_LR.fit(input_x,input_y)
best_para = grid_search_LR.best_params_
```

## Interview Experience Based Preparation
* Python sort() function time complexity is O(n log n) for average and worst cases.
* Nested For loop: n^2
  
```
nyc_data['neighbourhood_group']= nyc_data['neighbourhood_group'].astype("category").cat.codes
nyc_data['neighbourhood'] = nyc_data['neighbourhood'].astype("category").cat.codes
nyc_data['room_type'] = nyc_data['room_type'].astype("category").cat.codes
nyc_data.info()

nyc_data['price_log'] = np.log(nyc_data.price+1)

mean = nyc_model['reviews_per_month'].mean()
nyc_model['reviews_per_month'].fillna(mean, inplace=True)
nyc_model.isnull().sum()
corr=nyc_model.corr(method='pearson')
multicollinearity, V=np.linalg.eig(corr)
scaler = StandardScaler()
nyc_model_x = scaler.fit_transform(nyc_model_x)
X_train, X_test, y_train, y_test = train_test_split(nyc_model_x, nyc_model_y, test_size=0.3,random_state=42)
```

