# World Happiness Report Python Project
```python
# 1. Importing all dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```


```python
# 2. Loading dataset
data = pd.read_csv('/Users/verpuchinskiy/Downloads/world_happiness_2024.csv', sep=';', encoding_errors='ignore')
```


```python
# 3. Initial Exploration
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ranking</th>
      <th>Country</th>
      <th>Regional indicator</th>
      <th>Ladder score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>140</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>1,721</td>
      <td>2,93451</td>
      <td>0</td>
      <td>62</td>
      <td>0</td>
      <td>0,22638</td>
      <td>0,15383</td>
    </tr>
    <tr>
      <th>1</th>
      <td>86</td>
      <td>Albania</td>
      <td>Central and Eastern Europe</td>
      <td>5,3042</td>
      <td>6,71748</td>
      <td>0,57133</td>
      <td>74</td>
      <td>0,79892</td>
      <td>0,34403</td>
      <td>0,08517</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84</td>
      <td>Argelia</td>
      <td>Sub-Saharan Africa</td>
      <td>5,3635</td>
      <td>6,18327</td>
      <td>0,73652</td>
      <td>72</td>
      <td>0,28611</td>
      <td>0,22771</td>
      <td>0,34775</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>Argentina</td>
      <td>Latin America and Caribbean</td>
      <td>6,1881</td>
      <td>7,29612</td>
      <td>0,85449</td>
      <td>73</td>
      <td>0,78851</td>
      <td>0,21822</td>
      <td>0,1399</td>
    </tr>
    <tr>
      <th>4</th>
      <td>81</td>
      <td>Armenia</td>
      <td>Commonwealth of Independent States</td>
      <td>5,4549</td>
      <td>6,7441</td>
      <td>0,71415</td>
      <td>73</td>
      <td>0,75317</td>
      <td>0,12831</td>
      <td>0,3009</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (140, 10)




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 140 entries, 0 to 139
    Data columns (total 10 columns):
     #   Column                        Non-Null Count  Dtype 
    ---  ------                        --------------  ----- 
     0   Ranking                       140 non-null    int64 
     1   Country                       140 non-null    object
     2   Regional indicator            140 non-null    object
     3   Ladder score                  140 non-null    object
     4   GDP per capita                140 non-null    object
     5   Social support                140 non-null    object
     6   Healthy life expectancy       140 non-null    int64 
     7   Freedom to make life choices  140 non-null    object
     8   Generosity                    140 non-null    object
     9   Perceptions of corruption     140 non-null    object
    dtypes: int64(2), object(8)
    memory usage: 11.1+ KB



```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ranking</th>
      <th>Healthy life expectancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>140.0000</td>
      <td>140.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>70.5000</td>
      <td>70.828571</td>
    </tr>
    <tr>
      <th>std</th>
      <td>40.5586</td>
      <td>4.991998</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0000</td>
      <td>55.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35.7500</td>
      <td>67.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>70.5000</td>
      <td>72.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>105.2500</td>
      <td>75.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>140.0000</td>
      <td>81.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 4. Data Cleaning
data.isnull().sum()
```




    Ranking                         0
    Country                         0
    Regional indicator              0
    Ladder score                    0
    GDP per capita                  0
    Social support                  0
    Healthy life expectancy         0
    Freedom to make life choices    0
    Generosity                      0
    Perceptions of corruption       0
    dtype: int64




```python
data.duplicated().sum()
```




    np.int64(0)




```python
data.dtypes
```




    Ranking                          int64
    Country                         object
    Regional indicator              object
    Ladder score                    object
    GDP per capita                  object
    Social support                  object
    Healthy life expectancy          int64
    Freedom to make life choices    object
    Generosity                      object
    Perceptions of corruption       object
    dtype: object




```python
columns_to_change = ['Ladder score', 'GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']

for col in columns_to_change:
    data[col] = data[col].str.replace(',', '.').astype('float')
```


```python
data.dtypes
```




    Ranking                           int64
    Country                          object
    Regional indicator               object
    Ladder score                    float64
    GDP per capita                  float64
    Social support                  float64
    Healthy life expectancy           int64
    Freedom to make life choices    float64
    Generosity                      float64
    Perceptions of corruption       float64
    dtype: object




```python
data = data [['Ranking', 'Country', 'Regional indicator', 'Ladder score', 'GDP per capita', 'Healthy life expectancy', 'Social support', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
```


```python
columns_to_change = ['Social support', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
data[columns_to_change] = data[columns_to_change] * 10
```


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ranking</th>
      <th>Ladder score</th>
      <th>GDP per capita</th>
      <th>Healthy life expectancy</th>
      <th>Social support</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>140.0000</td>
      <td>140.000000</td>
      <td>140.000000</td>
      <td>140.000000</td>
      <td>140.000000</td>
      <td>140.000000</td>
      <td>140.000000</td>
      <td>140.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>70.5000</td>
      <td>5.530882</td>
      <td>6.441332</td>
      <td>70.828571</td>
      <td>7.017045</td>
      <td>7.188301</td>
      <td>3.650760</td>
      <td>2.679898</td>
    </tr>
    <tr>
      <th>std</th>
      <td>40.5586</td>
      <td>1.181221</td>
      <td>1.985895</td>
      <td>4.991998</td>
      <td>2.061883</td>
      <td>1.882227</td>
      <td>1.832604</td>
      <td>2.195626</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0000</td>
      <td>1.721000</td>
      <td>0.000000</td>
      <td>55.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35.7500</td>
      <td>4.631600</td>
      <td>5.033228</td>
      <td>67.000000</td>
      <td>5.699925</td>
      <td>6.108775</td>
      <td>2.273775</td>
      <td>1.189475</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>70.5000</td>
      <td>5.800600</td>
      <td>6.686325</td>
      <td>72.000000</td>
      <td>7.653500</td>
      <td>7.426250</td>
      <td>3.401000</td>
      <td>2.089000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>105.2500</td>
      <td>6.426050</td>
      <td>8.134660</td>
      <td>75.000000</td>
      <td>8.557850</td>
      <td>8.528775</td>
      <td>4.806350</td>
      <td>3.364725</td>
    </tr>
    <tr>
      <th>max</th>
      <td>140.0000</td>
      <td>7.740700</td>
      <td>10.000000</td>
      <td>81.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 5. Data Analysis
```


```python
# Univariate analysis
```


```python
# Ladder score distribution
bins = np.arange(0, 11, 1)
sns.histplot(data=data, x='Ladder score', bins=bins)
plt.ylabel('Number of countries')
plt.title('Ladder Score Distribution')
```




    Text(0.5, 1.0, 'Ladder Score Distribution')




    
![png](output_16_1.png)
    



```python
# GDP Per Capita Distribution
bins = np.arange(0, 11, 1)
sns.histplot(data=data, x='GDP per capita', bins=bins)
plt.ylabel('Number of countries')
plt.title('GDP Per Capita Distribution')
```




    Text(0.5, 1.0, 'GDP Per Capita Distribution')




    
![png](output_17_1.png)
    



```python
# Healthy Life Expectancy
bins = np.arange(0, 100, 5)
sns.histplot(data=data, x='Healthy life expectancy', bins=bins)
plt.ylabel('Number of countries')
plt.title('Healthy life expectancy Distribution')
```




    Text(0.5, 1.0, 'Healthy life expectancy Distribution')




    
![png](output_18_1.png)
    



```python
# Happiness Score based on region
data.groupby(by='Regional indicator')['Ladder score'].mean().sort_values(ascending=False)
```




    Regional indicator
    North America and ANZ                 6.927625
    Western Europe                        6.835591
    Latin America and Caribbean           6.143368
    Central and Eastern Europe            6.028571
    East Asia                             5.820771
    Southeast Asia                        5.603525
    Commonwealth of Independent States    5.535913
    Middle East and North Africa          5.164663
    Sub-Saharan Africa                    4.354306
    South Asia                            3.895700
    Name: Ladder score, dtype: float64




```python
# Bivariate analysis
```


```python
# Happiness Score and Healthy Life Expectancy relationship
plt.figure(figsize=(8, 6))
plt.title('Dependency of Happiness Score from Healthy Life Expectancy')
sns.scatterplot(data=data, x='Healthy life expectancy', y='Ladder score', hue='Regional indicator')
plt.legend(bbox_to_anchor=(1.55, 1), loc='upper right', borderaxespad=0)
```




    <matplotlib.legend.Legend at 0x14d08f200>




    
![png](output_21_1.png)
    



```python
sns.pairplot(data=data, vars=['Ladder score', 'GDP per capita', 'Healthy life expectancy', 'Social support', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'], hue='Regional indicator')
```




    <seaborn.axisgrid.PairGrid at 0x15bf53aa0>




    
![png](output_22_1.png)
    



```python
# Heatmap. Correlation between Happiness Score elements
```


```python
corr = data[['Ladder score', 'GDP per capita', 'Healthy life expectancy', 'Social support', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']].corr()
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ladder score</th>
      <th>GDP per capita</th>
      <th>Healthy life expectancy</th>
      <th>Social support</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ladder score</th>
      <td>1.000000</td>
      <td>0.768501</td>
      <td>0.758750</td>
      <td>0.813538</td>
      <td>0.644452</td>
      <td>0.129989</td>
      <td>0.451722</td>
    </tr>
    <tr>
      <th>GDP per capita</th>
      <td>0.768501</td>
      <td>1.000000</td>
      <td>0.827296</td>
      <td>0.726783</td>
      <td>0.414838</td>
      <td>-0.059443</td>
      <td>0.444252</td>
    </tr>
    <tr>
      <th>Healthy life expectancy</th>
      <td>0.758750</td>
      <td>0.827296</td>
      <td>1.000000</td>
      <td>0.702832</td>
      <td>0.406113</td>
      <td>0.002522</td>
      <td>0.393779</td>
    </tr>
    <tr>
      <th>Social support</th>
      <td>0.813538</td>
      <td>0.726783</td>
      <td>0.702832</td>
      <td>1.000000</td>
      <td>0.484731</td>
      <td>0.079675</td>
      <td>0.251008</td>
    </tr>
    <tr>
      <th>Freedom to make life choices</th>
      <td>0.644452</td>
      <td>0.414838</td>
      <td>0.406113</td>
      <td>0.484731</td>
      <td>1.000000</td>
      <td>0.223951</td>
      <td>0.344314</td>
    </tr>
    <tr>
      <th>Generosity</th>
      <td>0.129989</td>
      <td>-0.059443</td>
      <td>0.002522</td>
      <td>0.079675</td>
      <td>0.223951</td>
      <td>1.000000</td>
      <td>0.172513</td>
    </tr>
    <tr>
      <th>Perceptions of corruption</th>
      <td>0.451722</td>
      <td>0.444252</td>
      <td>0.393779</td>
      <td>0.251008</td>
      <td>0.344314</td>
      <td>0.172513</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(8, 6))
sns.heatmap(data=corr, annot=True)
plt.title('Happiness Score Correlation Heatmap')
```




    Text(0.5, 1.0, 'Happiness Score Correlation Heatmap')




    
![png](output_25_1.png)
    



```python

```
