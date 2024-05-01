import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

X = pd.read_csv(r'C:\data\Results_21Mar2022.csv')

X = X.drop(['mc_run_id'],axis=1)
X = X.drop(['grouping'],axis=1)

encoder = OrdinalEncoder()
columns_to_encode = ['sex', 'diet_group', 'age_group']
X[columns_to_encode] = encoder.fit_transform(X[columns_to_encode])

Y = X.copy()

Y = Y.drop(['sd_ghgs', 'sd_land', 'sd_watscar', 'sd_eut', 'sd_ghgs_ch4', 'sd_ghgs_n2o', 'sd_bio', 'sd_watuse', 'sd_acid'], axis=1)

correlation_matrix = Y.corr()
sns.clustermap(correlation_matrix, cmap = 'RdBu', fmt=".2f", annot = True, figsize = [20,10], center=0, cbar_kws={"ticks": [-1, 0, 1], "shrink": 0.5})
plt.title('Color Mapping', pad=10)
plt.subplots_adjust(top=0.95, bottom=0.2, right=0.9)
plt.show()


X = X.drop(['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut', 'mean_ghgs_ch4', 'mean_ghgs_n2o', 'mean_bio', 'mean_watuse', 'mean_acid'], axis=1)

correlation_matrix = X.corr()
sns.clustermap(correlation_matrix, cmap = 'RdBu', fmt=".2f", annot = True, figsize = [20,10], center=0, cbar_kws={"ticks": [-1, 0, 1], "shrink": 0.5})
plt.title('Color Mapping', pad=10)
plt.subplots_adjust(top=0.95, bottom=0.2, right=0.9)
plt.show()

