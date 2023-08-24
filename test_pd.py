# 96.12.26 ~ 23.06.20   6628
df = pd.read_parquet(f'data/{TICKER}.parquet')
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

df_sorted_descending = df.sort_values(by='date', ascending=True)
fig = px.line(df_sorted_descending, x='date', y=df_sorted_descending.index) # y='close')
fig.show()




from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

iris
# The data is stored in the 'data' attribute
iris_data = iris.data

# The target values are stored in the 'target' attribute
iris_target = iris.target

# The feature names are stored in the 'feature_names' attribute
feature_names = iris.feature_names

# The target names are stored in the 'target_names' attribute
target_names = iris.target_names

# Create a DataFrame from the data
import pandas as pd

df = pd.DataFrame(data=iris_data, columns=feature_names)
df['target'] = iris_target
iris = df.copy()
# Display the first few rows of the DataFrame
print(df.head())
