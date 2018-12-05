import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# import ing the iris dataset
dataset = pd.read_csv('input/Iris.csv')

# looking into the first few rows of the dataset
dataset.head()

# looking gow much samples we got for each species => each one is 50 samples
dataset["Species"].value_counts()

# regular scatterplot for Sepal Length by Sepal Width spread by Species in color
sctPlt = sns.FacetGrid(dataset, hue="Species", height=5) \
            .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
            .add_legend()
sctPlt = plt.show()

# boxplot by Petal Length for each Species
bxPLt = sns.boxplot(x="Species", y="PetalLengthCm", data=dataset)
bxPLt = plt.show()


# boxplot of Petal Length for each Species with scatter plot jittered on top of it
bSPlt = sns.boxplot(x="Species", y="PetalLengthCm", data=dataset)
bSPlt = sns.stripplot(x="Species", y="PetalLengthCm", data=dataset, jitter=True, edgecolor="gray")
bSPlt = plt.show()

# violinplot Petal Length for each Species - the violinPlot replaces 
# the jittered scatterplot and shows where the density is high
vPlt = sns.violinplot(x="Species", y="PetalLengthCm", data=dataset, height=6)
vPlt = plt.show()

# distribution of Petal Length on each Species - called kdePlot
kdPlt = sns.FacetGrid(dataset, hue="Species", height=6) \
            .map(sns.kdeplot, "PetalLengthCm") \
            .add_legend()
kdPlt = plt.show()

# correlogram for all the features - shows correlation of each feature with others
hmPlt = sns.heatmap(dataset.drop("Id", axis=1).corr(), annot = True, cmap = 'cubehelix_r')
hmPlt = plt.show()

# pair plot of features interacting with eachother - same as correlogram but scatterploted
# in the main diagonal we can see the distributions of each feature for each species
prPlt = sns.pairplot(dataset.drop("Id", axis=1), hue="Species", height=2)
prPlt = plt.show()

# plot that shows each sample as a line that connected by features - Species spread by colors
# in future might be needed to change the import to 'pandas.tools.plotting.parallel_coordinates'
from pandas.tools.plotting import parallel_coordinates
plPlt = parallel_coordinates(dataset.drop("Id", axis=1), "Species")
plPlt = plt.show()

# last plot - radviz - is kind of gravity plot that has points which are samples and 
# it shows their gravity with regard to the feature
# might be useful to spot a nonlinear relationships
# in future will need to change import to 'pandas.plotting.radviz'
from pandas.tools.plotting import radviz
rdPlt = radviz(dataset.drop("Id", axis=1), "Species")
rdPlt = plt.show()

