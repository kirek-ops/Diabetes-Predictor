import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Data
data_path = 'data/diabetes.csv'
df = pd.read_csv(data_path)

# Try to find correlation 
sns.heatmap(df.corr() , annot = True)
plt.show()