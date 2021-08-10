import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression 
from sklearn import metrics  

file_data = pd.read_csv('task1_data.csv')
hours = file_data.iloc[:,:-1].values
scores = file_data.iloc[:,1].values
test_score_data = [[1.5], [3.2], [7.4], [2.5], [5.9]]
print(file_data.head())

plt.scatter(file_data['Hours'],file_data['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Relationship between Hours studied and Score obtained')
plt.legend(['Score'])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(hours, scores, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Model Trained.")

regression_line = regressor.coef_*hours + regressor.intercept_

plt.scatter(hours, scores)
plt.plot(hours, regression_line)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Regression Line')
plt.show()

test_reading = [[9.25]]
pred = regressor.predict(test_reading)
print('Prediction at '+ str(test_reading[0][0]) + ' is '+ str(pred[0]))

pred_data = regressor.predict(test_score_data)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_data)) 