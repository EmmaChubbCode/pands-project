# Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.
#  data has been downloaded from https://archive.ics.uci.edu/dataset/53/iris which includes the option to importdata using python.

# ucimlrepo is a package for importing datasets from the the UC Irvine Machine Learning Repository.
# See: https://github.com/uci-ml-repo/ucimlrepo 
from ucimlrepo import fetch_ucirepo 

# fetch the ds. the ID specifies whic of the UCI datasets you want.
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# metadata contains details of the dataset including its main characterisics, shape, info on missing data, and relevant links (e.g. where to find raw data) 
# the meta data also contains detailed additional information including text descriptions of variables, funding sources, and the purpose of the data, 
print(iris.metadata) 

print(iris.metadata.additional_info)

# variable information 
print(iris.variables)

iris_df = iris.data

print(iris_df)

print(iris.additional)