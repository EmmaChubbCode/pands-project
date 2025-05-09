# First, import necessary libraries for importing data and whatever analysis follows
# will put these in a requirements file later
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import datasets
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder as le
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statistics

# ucimlrepo is a package for importing datasets from the the UC Irvine Machine Learning Repository.
# See: https://github.com/uci-ml-repo/ucimlrepo     
from ucimlrepo import fetch_ucirepo

# fetch the data. the ID specifies which of the UCI datasets you want.
iris = fetch_ucirepo(id=53) 

# metadata contains details of the dataset including its main characterisics, shape, info on missing data, and relevant links (e.g. where to find raw data) 
# the meta data also contains detailed additional information including text descriptions of variables, funding sources, and the purpose of the data, 
print(iris.metadata) 

# here we will begin writing the output to a file
# first create a file name 'irisis_project.txt' which we will alias as f
# the 'wt' option means we are writing to a text file.
# the following resource was used for writing this code: https://www.geeksforgeeks.org/reading-writing-text-files-python/#write-to-text-file-in-python 
with open('iris_project.txt', 'wt') as f:
    print(iris.metadata, file=f) # print the metadata to the file

# lets take the data and save it to a variable called iris
iris = iris.data

# now that iris is where we've stored the data, print iris to see what it contains
print(iris)

# append the above shown above to my iris_project.txt file
# using the append method found: https://realpython.com/read-write-files-python/#appending-to-a-file 
with open('iris_project.txt', 'at') as f:
    print(iris, file=f) # print the data to the file

# look at the features of the data. you can see the columns represent sepal length, sepal width, petal length, and petal width.
print(iris.features)

# append to the file
with open('iris_project.txt', 'at') as f:
    print(iris.features, file=f)

# the targets are labels for the data. in this case, they are the species of iris flower (setosa, versicolor, virginica).
print(iris.targets)

# and save the targets to the file
with open('iris_project.txt', 'at') as f:
    print(iris.targets, file=f)

# I would like to have both the targets and features in one dataframe to make my analysis and code easier. 
# I put each into variables called X and y in preparation for concatenation.
X = iris.features 
y = iris.targets

# i then used these two variables to create a new dataframe called iris_df.
# we'll use the pandas function concat to do this. we'll specify we're joining on axis=1, which means we're joining on the columns. 
# see: https://pandas.pydata.org/docs/user_guide/merging.html#joining-logic-of-the-resulting-axis 
iris_df = pd.concat([X, y], axis=1)

# return top 5 rows
# see: https://www.w3schools.com/python/pandas/ref_df_head.asp#:~:text=The%20head()%20method%20returns,a%20number%20is%20not%20specified.&text=Note%3A%20The%20column%20names%20will,addition%20to%20the%20specified%20rows.
iris_df.head()

# append to the file
with open('iris_project.txt', 'at') as f:
    print(iris_df.head(), file=f) # print the data to the file

# return bottom 5 rows.
# https://www.w3schools.com/python/pandas/ref_df_tail.asp#:~:text=The%20tail()%20method%20returns,a%20number%20is%20not%20specified.
iris_df.tail()

# append to the file
with open('iris_project.txt', 'at') as f:
    print(iris_df.tail(), file=f) # print the data to the file

# next check the shape of iris data, meaning how many rows and how many columns.
iris_df.shape

# append to the file
with open('iris_project.txt', 'at') as f:
    print(iris_df.shape, file=f) # print the data to the file

# dtypes returns the data types of each column in the dataframe.
# see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dtypes.html
iris_df.dtypes

# append the types to the file
with open('iris_project.txt', 'at') as f:
    print(iris_df.dtypes, file=f)

# append the types to the file
with open('iris_project.txt', 'at') as f:
    print(iris_df.dtypes, file=f)

# we can check for nulls by combining the ifnull function with the sum function
# see: https://www.w3schools.com/python/pandas/ref_df_isnull.asp 
# and https://www.w3schools.com/python/pandas/ref_df_sum.asp
print(iris_df.isnull().sum())

# save null details to the file
with open('iris_project.txt', 'at') as f:
    print(iris_df.isnull().sum(), file=f)

# Describe the data set. This will show basic descriptive statistics for each column in the dataframe.
# This includes the count, mean, standard deviation, min, max, and 25th, 50th, and 75th percentiles.
# see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html 
iris_df.describe()

# append describe output to the file
with open('iris_project.txt', 'at') as f:
    print(iris_df.describe(), file=f) # print the data to the file

# the skew function will show us the skewness of the data. the skewness of a measure of how distributed the data is around the mean. 
# see: https://www.datacamp.com/tutorial/understanding-skewness-and-kurtosis 
# i want it for each column so im going to use for loop to save time. see: https://statisticsglobe.com/iterate-over-columns-pandas-dataframe-python 

# for each column in iris df, calculate the skewness and then print it out.  
for column in iris_df:
   if column != 'class': # first check if the column is not the class column. that has strings so won't work - learned this from earlier error. 
    skew = iris_df[column].skew()
    print (f"Skewness of {column}: {skew}")

# save skewness to the file
with open('iris_project.txt', 'at') as f:
    for column in iris_df:
        if column != 'class':
            skew = iris_df[column].skew()
            print (f"Skewness of {column}: {skew}", file=f)

# Similarly, we can check the data for kurtosis. According to data camp, "kurtosis focuses more on the height. It tells us how peaked or flat our normal (or normal-like) distribution is. 
# see https://www.datacamp.com/tutorial/understanding-skewness-and-kurtosis
# for each column in iris df, calculate the skewness and then print it out.  
for column in iris_df:
   if column != 'class': # first check if the column is not the class column. that has strings so won't work - learned this from earlier error. 
        kurtosis = iris_df[column].kurtosis()
        print (f"Kurtosis of {column}: {kurtosis}")

# save kurtosis to the file
with open('iris_project.txt', 'at') as f:
    for column in iris_df:
        if column != 'class':
            kurtosis = iris_df[column].kurtosis()
            print (f"Kurtosis of {column}: {kurtosis}", file=f)

# The class column is a string variable and therefore we cannot calculate mean, median, skewness, or kurtosis as we did above. However, we can count the occurence of each value.
# the value_counts function will return a series containing counts of unique values. 
# see: https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html 
iris_df['class'].value_counts()

# append the counts to the file
with open('iris_project.txt', 'at') as f:
    print(iris_df['class'].value_counts(), file=f)

# put the columns into their own variables so i can do a histogram for each. give them short names for sake of writing code later.
sl = iris_df['sepal length']
sw = iris_df['sepal width']
pl = iris_df['petal length']
pw = iris_df['petal width']

# for documentation on hist function, see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
# for example of hist using iris see: https://www.geeksforgeeks.org/box-plot-and-histogram-exploration-on-iris-data/ 
# For example of plotting multiple hists on one plot see: https://www.geeksforgeeks.org/plotting-histogram-in-python-using-matplotlib/  

# Creating subplots with multiple histograms. i have 4 things to display so i'm doing a 2x2 plot.
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 4))

# these axes figures dictate where on the plot the particular subplot will appear (e.g. 0,0 = first row, first column.)
# specify your datasource, number of bins, colour of bars, and colour of outline.
axes[0,0].hist(sl, bins=20, color='Yellow', edgecolor='black')
axes[0,0].set_title('Sepal Length in cm')
 
axes[0,1].hist(sw, bins=20, color='Pink', edgecolor='black')
axes[0,1].set_title('Sepal Width in cm')

axes[1,0].hist(pl, bins=20, color='Blue', edgecolor='black')
axes[1,0].set_title('Petal Length in cm')

axes[1,1].hist(pw, bins=20, color='Red', edgecolor='black')
axes[1,1].set_title('Petal Width in cm')

# Adding labels and title. initial error iterating over each subplot individually- code adjusted by Microsoft Co-Pilot.
for ax in axes.flat:
    ax.set_xlabel('Centimeters')
    ax.set_ylabel('Counts')

# Adjusting layout for better spacing. without this all the titles start overlapping.
plt.tight_layout()

# Saving the plot as a png file
# see: https://www.geeksforgeeks.org/how-to-save-a-plot-to-a-file-using-matplotlib/ 
plt.savefig("histograms.png")

# Display the figure
plt.show()

# using the statistics library, calculate the mode for each of the four columns.
statistics.mode(sl), statistics.mode(sw), statistics.mode(pl), statistics.mode(pw)

# append modes to the file
with open('iris_project.txt', 'at') as f:
    print(statistics.mode(sl), file=f)
    print(statistics.mode(sw), file=f)
    print(statistics.mode(pl), file=f)
    print(statistics.mode(pw), file=f)

# to make a boxplot in matplot you have to drop any non-numerical data. my data set has species. lets drop that for a boxplot friendly version of the df.
# see: https://www.nickmccullum.com/python-visualization/boxplot/ 
iris_boxplot = iris_df.drop(columns=['class'])

# Create figure, axis.
fig, ax = plt.subplots()

# before passing the data to the box plot function, i'm going to create flierprops as noted here: https://matplotlib.org/3.1.3/gallery/statistics/boxplot.html 
flierprops = dict(marker='o', markerfacecolor='green', markersize=12, linestyle='none')

# Create boxplot. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html 
ax.boxplot(iris_boxplot, flierprops=flierprops)

# # Set names of x-axis ticks.
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html
ax.set_xticks([1, 2, 3, 4], ["sepalLength", "sepalWidth", "petalLength", "petalWidth"], fontsize=10)
ax.set_title("Side-by-Side Boxplots of Iris Features", fontsize=16)

# save the boxplot as a png file
plt.savefig("boxplot.png")

# that was a helpful overview of the data but i also want to check the individual class types and their distribution. To view individual class types, we can use the seaborn library.
# see: https://seaborn.pydata.org/generated/seaborn.boxplot.html

# how to set the title: https://how.dev/answers/how-to-add-a-title-to-a-seaborn-plot 
# im putting the different species across the x axis and using the sepal length column to plot my y. this is should give me the sepal length for all of the classes if my dataset is working how i want it to.
plt.figure(figsize=(10, 6))
sns.boxplot(x="class", y="sepal length", data=iris_df).set_title("Compare the Distributions of Sepal Length")

# save the boxplot as a png file
plt.savefig("boxplot_sepallength.png")

# repeat for other features
plt.figure(figsize=(10, 6))
sns.boxplot(x="class", y="sepal width", data=iris_df).set_title("Compare the Distributions of Sepal Width")

# save the boxplot as a png file
plt.savefig("boxplot_sepalwidth.png")

## repeat for other features
plt.figure(figsize=(10, 6))
sns.boxplot(x="class", y="petal width", data=iris_df).set_title("Compare the Distributions of Petal Width")

# save the boxplot as a png file
plt.savefig("boxplot_petalwidth.png")

# repeat for other features
plt.figure(figsize=(10, 6))
sns.boxplot(x="class", y="petal length", data=iris_df).set_title("Compare the Distributions of Petal Length")

# save the boxplot as a png file
plt.savefig("boxplot_petallength.png")

# set the styling of the figure. see: https://seaborn.pydata.org/generated/seaborn.set_style.html#seaborn-set-style 
plt.figure(figsize=(10, 6))
sns.set_style('darkgrid')

# see: https://seaborn.pydata.org/generated/seaborn.pairplot.html 
sns.pairplot(iris_df, hue="class")

# save the pairplot as a png file
plt.savefig("pairplot.png")

# see the following on how to structure the code: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html#pandas-dataframe-corr 
# When you don't specify parameters, this method defaults to Pearson correlation which is appropriate when using continuous and normally distributed data.
# see for more detail: https://docs.vultr.com/python/third-party/pandas/DataFrame/corr 
# you can only do the correlation on numeric data so i'll use the boxplot dataset i prepared before.
iris_boxplot.corr()

# save correlation to the file
with open('iris_project.txt', 'at') as f:
    print(iris_boxplot.corr(), file=f)

# use seaborn's heatmap function. See: https://seaborn.pydata.org/generated/seaborn.heatmap.html 
# cmap is optional parameter indicating how you want values to map to colour values.
# annot is an optional parameter determining if you want the heatmap annotated, and in this case i do.
# i've also added lines between cells to make it a bit more readable and less of a solid block of colour. 
# you can do loads as seen here: https://python-graph-gallery.com/92-control-color-in-seaborn-heatmaps/ 
sns.heatmap(iris_boxplot.corr(),cmap="crest", annot=True, linewidth=.5)

# save the heatmap as a png file
plt.savefig("heatmap.png")

#  create figure and axis to plot onto
# following stucture of code in lecture and  here: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
fig, ax = plt.subplots()

# seaborn enables the colour coding of classes in the scatter plot very easily, that's the main reason I'm using it over matplotlib.
# code structure for plotting graph found here: https://seaborn.pydata.org/generated/seaborn.scatterplot.html 
# and https://www.geeksforgeeks.org/plotting-graph-for-iris-dataset-using-seaborn-and-matplotlib/ 
# Hue is a grouping variable that will produce points with different colors. It can be either categorical or numeric.
# palette is like colour map in matplot and you can choose your selection from those listed here: http://matplotlib.org/stable/users/explain/colors/colormaps.html 
sns.scatterplot(data=iris_df, x='petal width', y='petal length', hue='class', palette= 'viridis') 

# create some labels for the axes
ax.set_xlabel('petal length (cm)')
ax.set_ylabel('petal width (cm)')

#add title to chart
plt.title('Scatter Plot of Petal Width vs. Petal Length')

# save the scatterplot as a png file
plt.savefig("scatterplot.png")

#show the plt
plt.show()

# https://realpython.com/linear-regression-in-python/#simple-linear-regression-with-scikit-learn 
# one point above states linearregression() requires x to be a 2D array. 
# i initially got errors because this wasn't the case. the reshape(-1,1) below basically changes the x values from being a 1D array with 150 instances to a 2D array with 150 rows and 1 column.
# to use numpy reshape on a pandas df you have to use values as well which i learned from: https://stackoverflow.com/questions/14390224/how-to-reshape-a-pandas-series#:~:text=you%20can%20directly%20use%20a,convert%20DataFrame%20to%20numpy%20ndarray 

# set out the data I'll input to the regression function
x = iris_df["petal length"].values.reshape((-1, 1))
y = iris_df["petal width"]

# create a model that fits x points and y points using linear regression    
# after errors i realise i had to reshape 
model = LinearRegression().fit(x, y)

# get the r squared score from the model
r_sq = model.score(x, y)
# print the result
print(f"coefficient of determination: {r_sq}")

# save the r squared score to the file
with open('iris_project.txt', 'at') as f:
    print(f"coefficient of determination: {r_sq}", file=f)

# round my r square value for the next task
r_sq = round(r_sq,2)

x = iris_df['petal length']
y = iris_df['petal width']

# Perform linear fit
coefficients = np.polyfit(x, y, 1)
print("Linear Fit Coefficients:", coefficients)

# Create polynomial function
p = np.poly1d(coefficients)

#plot this on a scatter plot, add labels of the data points, then plot the linear fit line.
plt.scatter(x, y, label='Data Points')
plt.plot(x, p(x), label='Linear Fit', color='red')
# instead of messing with the axis im trying to annotate inside the plot
# im adapting code from here but using python plain text formatting because of exposure to that style via Andrew: https://matplotlib.org/stable/users/explain/text/annotations.html#basic-annotation 
# the xy argument dictates placement
plt.annotate(f'R^2 = {r_sq:.3f}', xy=(5, 0.5),fontsize=11, color='black')
# create some labels for the axes
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Linear Fit of Petal Width vs. Petal Length')

# save the scatterplot as a png file
plt.savefig("linearfit.png")

# make the legend for the plot
plt.legend()
# show the plot
plt.show()


