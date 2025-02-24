import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
# print(type(ls))
# print(ls)
x = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values

# print(x)

ls.plot(kind = 'scatter', grid = True, x= 'GDP per capita (USD)', y = 'Life satisfaction')
plt.axis([23500, 62500, 4, 9])
plt.show()

# model = LinearRegression()
# model.fit(x,y)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)

x_new = [[37655.2]]
print(model.predict(x_new))