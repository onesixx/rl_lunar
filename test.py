# Exploring Titanic dataset with copilot

import pandas as pd
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
df.head()

# Let's have an overview of the dataset by printing the first 5 rows
# and the last 5 rows of the dataset
df.head()
df.tail()


# q: what is the meaning of Parch?
# a: Number of Parents/Children Aboard

# Let's compute how many people survived
df['Survived'].value_counts()

# compute how many passengers have not survived
len(df[df.Survived == 0])

# look at the correlation between the columns Survived and Pclass
# and visulize the numbers of people per each class survived or not with a catplot using plotly



# q: how can I use plotly to plot a catplot with seaborn?
# a:
# a: plotly.express.cat is not a function, but a class.

import seaborn as sns
sns.catplot(x='Survived', col='Pclass', kind='count', data=df)

df.corr()['Survived'].sort_values().plot(kind='bar')

