# Time-Series
Applying machine learning methods to predict time series

Даны 100 рядов с 1000 значениями в каждом.
Дисперсия шума ![](http://latex.codecogs.com/gif.latex?%5Csigma%5E2%3D1), дисперсия угла наклона тренда ![](http://latex.codecogs.com/gif.latex?%5Csigma_%7Ba%7D%5E%7B2%7D), средняя частота смены тренда 1 раз за 200 точек (![](http://latex.codecogs.com/gif.latex?%5Clambda%3D200)).
Ошибка излома оценивается как дисперсия суммы 2 нормальных с.в. (независимых углов наклона с обоих сторон от излома) с дисперсией 0.25, т.е. равна 0.5. Чтобы оценить суммарную ошибку для сложного тренда надо ошибку излома умножить на ожидание к-во изломов, т.е. ![](https://latex.codecogs.com/gif.latex?n/%5Clambda), где n --- длина предсказываемого ряда ряда, и, затем, усреднить, т.е. разделить на n. В итоге получим формулу ![](https://latex.codecogs.com/gif.latex?2%5Csigma_a%5E2/%5Clambda).

### Bounds:
series type | lower bound | upper bound
--- | --- | ---
linear trend | 0 | ![](https://latex.codecogs.com/gif.latex?%5Csigma_a%5E2%3D0.25)
Brownian motion | ![](http://latex.codecogs.com/gif.latex?%5Csigma%5E2%3D1) | -
linear trend with Brownian motion | ![](http://latex.codecogs.com/gif.latex?%5Csigma%5E2%3D1) | ![](https://latex.codecogs.com/gif.latex?%5Csigma%5E2&plus;%5Csigma_a%5E2%3D1.25)
difficult trend | ![](https://latex.codecogs.com/gif.latex?2%5Csigma_a%5E2/%5Clambda%3D0.0025) | ![](https://latex.codecogs.com/gif.latex?%5Csigma_a%5E2&plus;2%5Csigma_a%5E2/%5Clambda%3D0.2525)
difficult trend with Brownian motion | ![sdf](https://latex.codecogs.com/gif.latex?%5Csigma%5E2&plus;2%5Csigma_a%5E2/%5Clambda%3D1.0025) |  ![](https://latex.codecogs.com/gif.latex?%5Csigma%5E2&plus;%5Csigma_a%5E2&plus;2%5Csigma_a%5E2/%5Clambda%3D1.2525)

### Salnikov Dmitry
Есть 2 способа предсказания временных рядов по типу данных: по изначальным значениям и по их разностям ![](https://latex.codecogs.com/gif.latex?x_%7Bi&plus;1%7D-x_i). В случае линейных трендов предсказание по разностям будет оптимальным, им мы и будем пользоваться.

Рассмотрим простую и регуляризованную Lasso линейные регрессии, простое дерево и градиентный бустинг для предсказания последних 300 точек каждого ряда. Сравним ээфективности методов, а заодно посмотрим, как наличие интерсепта в регрессиях влияет на результат.

##### Simple trend
Все методы имеют нулевую ошибку.
Для предсказания в каждом методе использовалось только последнее значение разности.

##### Brownian motion
Кроссвалидация по окнам 2,4,8,12,16,20.
conf int: ![](https://latex.codecogs.com/gif.latex?2sd%28x%29/%5Csqrt%7Bn%7D), n=100 - ширина доверительного интервала в 1 сторону.
Лучшее предсказание --- по предыдущему значению исходных данных или просто нулями для разностей;

name | train | test | conf int
--- | --- | --- | ---
Best | - | 1.0107 | -
Lin regr | 0.991 | 1.015 | 0.018
Lin regr with intercept | 0.993 | 1.017 | 0.018
Lasso | 0.985 | 1.012 | 0.018
Lasso with intercept | 0.986 | 1.014 | 0.018
Tree | 1.010 | 1.048 | 0.020
XGBoost | 0.991 | 1.024 | 0.019

##### Simple trend + Brownian motion
Кроссвалидация по окнам 2,4,8,12,16,20.
Худшее предсказание --- по предыдущему значению исходных данных;
Лучшее --- по предыдущей разности с вычетом среднего значения по всем предыдущим элементам.

name | train | test | conf int
--- | --- | --- | ---
Best | - | 1.007 | -
Worst | - | 1.221 | -
Lin regr | 1.051 | 1.063 | 0.018
Lin regr with intercept | 1.006 | 1.015 | 0.018
Lasso | 1.044 | 1.056 | 0.018
Lasso with intercept | 1.000 | 1.012 | 0.018
Tree | 1.027 | 1.043 | 0.019
XGBoost | 1.005 | 1.022 | 0.018

##### difficult trend
Предсказываем по предыдущей разности.
Лучшее предсказание равно предыдущей разности.

name | train | test | conf int
--- | --- | --- | ---
Best | - | 0.00226 | -
Lin regr | 0.003 | 0.002 | 0.001
Lin regr with intercept | 0.056 | 0.002 | 0.001
Lasso | 0.003 | 0.002 | 0.001
Lasso with intercept | 0.026 | 0.002 | 0.001
Tree | 0.061 | 0.057 | 0.034
XGBoost | 0.056 | 0.056 | 0.034

##### difficult trend with Brownian motion
Окна от 2 до 20 с шагом 1 и от 25 до 115 с шагом 5.
Лучшее предсказание --- по предыдущей разности с вычетом среднего значения, посчитанного по последним n элементам, n подбирается кроссвалидацией.
Худшее --- по предыдущему элементу.

name | train | test | conf int
--- | --- | --- | ---
Best | - | 1.027 | -
Worst | - | 1.258 | -
Lin regr | 1.085 | 1.091 | 0.053
Lin regr with intercept | 1.112 | 1.150 | 0.053
Lasso | 1.062 | 1.090 | 0.020
Lasso with intercept | 1.083 | 1.149 | 0.049
Tree | 1.171 | 1.291 | 0.064
XGBoost | 1.099 | 1.181 | 0.023


### Egor Orbidan

#### Linear trend

name | diff | intercept| train | test 
--- | --- | --- | --- | ---
Linear Regression | True | True | 1.137317e-28 | 6.000043e-28
Linear Regression | True | False | 1.358522e-28 | 7.575085e-28
Linear Regression | False | True | 8.188263e-28 | 3.295571e-27
Linear Regression | False | False | 2.158090e-27 | 8.851460e-27
Lasso | True | True | 1.133105e-28 | 6.015045e-28
Lasso | True | False | 3.955797e-28 | 1.933942e-27
Lasso | False | True | 3.723566e-28 | 1.256569e-27
Lasso | False | False | 8.859158e-03 | 3.640814e-02
Ridge | True | True | 1.134711e-28 | 5.992114e-28
Ridge | True | False | 1.203431e-28 | 6.625225e-28
Ridge | False | False | 9.683097e-28 | 4.830781e-27
Ridge | False | False | 2.790511e-27 | 1.188342e-26

##### Brownian motion

name | diff | intercept| train | test 
--- | --- | --- | --- | ---
Linear Regression | True | True | 0.993243 | 1.017127
Linear Regression | True | False | 0.991034 | 1.015029	
Linear Regression | False | True | 0.973410 | 1.044813
Linear Regression | False | False | 0.977444 | 1.031223
Lasso | True | True | 0.983313 | 1.013945
Lasso | True | False | 0.985822 | 1.011624
Lasso | False | True | 0.975484	 | 1.049827
Lasso | False | False | 1.016391 | 1.077941
Ridge | True | True | 0.972734 | 1.015794
Ridge | True | False | 0.970044 | 1.014919
Ridge | False | False | 0.974569 | 1.044066
Ridge | False | False | 0.978227 | 1.028636	

##### Linear trend + Brownian motion

name | diff | intercept| train | test 
--- | --- | --- | --- | ---
Linear Regression | True | True | 1.006181 | 1.015163
Linear Regression | True | False | 1.017119 | 1.062211	
Linear Regression | False | True | 0.991068 | 1.036579
Linear Regression | False | False | 0.994755 | 1.065599
Lasso | True | True | 0.995646 | 1.012500
Lasso | True | False | 0.987633 | 1.048501
Lasso | False | True | 2.163391	 | 2.431317
Lasso | False | False | 2.059733 | 2.198690
Ridge | True | True | 0.985847 | 1.014413
Ridge | True | False | 0.970047 | 1.032641
Ridge | False | False | 0.990298 | 1.037995	
Ridge | False | False | 0.991251 | 1.066915	

##### Difficult trend 

name | diff | intercept| train | test 
--- | --- | --- | --- | ---
Linear Regression | True | True | 0.002604 | 0.100542	
Linear Regression | True | False | 0.002716 | 0.003495	
Linear Regression | False | True | 0.002622 | 0.339689
Linear Regression | False | False | 0.002639 | 0.004418
Lasso | True | True | 0.002894 | 0.012319
Lasso | True | False | 0.003301 | 0.002703
Lasso | False | True | 1.248994	 | 5.607846
Lasso | False | False | 3.218304 | 23.458689
Ridge | True | True | 0.002858 | 0.008180
Ridge | True | False | 0.002798 | 0.002604
Ridge | False | False | 0.003352 | 0.402696	
Ridge | False | False | 0.002666 | 0.003385	

##### Difficult trend with Brownian motion

name | diff | intercept| train | test 
--- | --- | --- | --- | ---
Linear Regression | True | True | 1.027051 | 1.144932	
Linear Regression | True | False | 1.039117 | 1.089997	
Linear Regression | False | True | 1.028862 | 1.262916
Linear Regression | False | False | 1.021539 | 1.151785
Lasso | True | True | 1.038217 | 1.169991
Lasso | True | False | 1.031077 | 1.094633
Lasso | False | True | 2.786531	 | 5.688668
Lasso | False | False | 3.298420 | 6.176644	
Ridge | True | True | 1.025914 | 1.151735
Ridge | True | False | 1.006670 | 1.074331
Ridge | False | False | 1.028872 | 1.268758	
Ridge | False | False | 1.022361 | 1.152730
DT | True | None | 1.147730 | 1.332669
DT | False | None | 2.872477 | 12645.238931
KNN | True | None | 1.145700 | 1.353376
KNN | False | None | 2.211956 | 12562.409741
