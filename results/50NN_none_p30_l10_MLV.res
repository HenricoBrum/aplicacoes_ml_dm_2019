cluster  1   ['NOV', 'AEE', 'VZ', 'EMR', 'MPC', 'WFC', 'NUE', 'VIAB', 'ABT', 'CPB']
cluster  2   ['AGN', 'ROK', 'FDX', 'WAT', 'PH', 'ADS', 'AVGO', 'NOC', 'WYNN', 'HUM']
cluster  3   ['GE', 'BK', 'CSX', 'NI', 'NEM', 'AOS', 'HRL', 'HCP', 'GM', 'UDR']
cluster  4   ['AAP', 'MCO', 'INTU', 'ACN', 'DLR', 'VNO', 'CAT', 'FRT', 'DIS', 'CXO']
cluster  5   ['DHR', 'IR', 'COF', 'MSI', 'QCOM', 'EXR', 'TSCO', 'ADSK', 'MDT', 'D']
reading files...
(61384, 31)
(61384,)
evaluating cluster  0
(9824, 30)
fit predict SVR...
fit predict MLP...
SVR MSE:  61.37766975726776
SVR POCID:  100.0
SVR EXACT:  0
SVR THEIL'S U:  9.64559604472393
MLP MSE:  0.9194725090709426
MLP POCID:  100.0
MLP EXACT:  0
MLP THEIL'S U:  1.2825968326005748
BASELINE MSE:  0.5520047259934853
BASELINE EXACT:  22
44.06   44.66516152507333   44.00577328634233   43.37
56.04   54.225428105140566   54.586512753873066   54.35
35.93   35.835019929674836   36.435355991029226   35.92
61.49   61.63174870103647   61.48029205784276   61.02
24.41   47.7762205719052   23.80359160408798   23.2
evaluating cluster  1
(9824, 30)
fit predict SVR...
fit predict MLP...
SVR MSE:  3702.4288434696236
SVR POCID:  100.0
SVR EXACT:  0
SVR THEIL'S U:  23.48143066342525
MLP MSE:  10.461090306639157
MLP POCID:  100.0
MLP EXACT:  0
MLP THEIL'S U:  1.1251049070002346
BASELINE MSE:  8.17575970327769
BASELINE EXACT:  11
102.84   152.2926595737382   104.40959469302618   103.52
311.76   152.7248995157855   311.80448724209276   310.93
187.26   152.72514145692773   186.06560944790562   185.06
73.11   149.95173725808647   74.61178495086362   74.01
283.89   152.7249317030601   281.42347060559115   280.74
evaluating cluster  2
(9824, 30)
fit predict SVR...
fit predict MLP...
SVR MSE:  9.773885505208414
SVR POCID:  100.0
SVR EXACT:  0
SVR THEIL'S U:  6.379373217333086
MLP MSE:  0.8570954610024143
MLP POCID:  100.0
MLP EXACT:  0
MLP THEIL'S U:  1.8951420049531342
BASELINE MSE:  0.2491915159649838
BASELINE EXACT:  28
43.64   44.13243783114606   42.93582786723632   43.83
31.05   32.165746295266445   31.76817759044795   32.18
16.67   25.38015346536188   15.598932844038545   15.76
32.11   31.70342523445612   31.166036964523265   31.77
38.22   37.85461329943129   36.91195995886233   37.89
evaluating cluster  3
(9824, 30)
fit predict SVR...
fit predict MLP...
SVR MSE:  657.9897803168084
SVR POCID:  100.0
SVR EXACT:  0
SVR THEIL'S U:  15.639948940050209
MLP MSE:  3.1421574132876806
MLP POCID:  100.0
MLP EXACT:  0
MLP THEIL'S U:  1.0990893574065483
BASELINE MSE:  2.5929510431636813
BASELINE EXACT:  14
81.64   99.19352093507727   82.54890520319763   82.15
103.09   103.52447706068152   104.99665139939908   104.05
119.66   103.10925962632865   116.6167537119858   117.34
157.41   102.01241396760892   158.023582068365   158.24
115.21   104.57454838341309   114.49767361676602   113.56
evaluating cluster  4
(9811, 30)
fit predict SVR...
Traceback (most recent call last):
  File "process_dataset.py", line 311, in <module>
    main(path, period, transform, attribute, fold)
  File "process_dataset.py", line 245, in main
    svr_pred = classify(X_train[:,1:], Y_train.ravel(), X_test[:,1:], clf=SVR(C=1.0, epsilon=0.2))
  File "process_dataset.py", line 113, in classify
    clf.fit(X_train,Y_train)
  File "/home/rico/.local/lib/python3.6/site-packages/sklearn/svm/base.py", line 147, in fit
    y = self._validate_targets(y)
  File "/home/rico/.local/lib/python3.6/site-packages/sklearn/svm/base.py", line 233, in _validate_targets
    return column_or_1d(y, warn=True).astype(np.float64, copy=False)
ValueError: setting an array element with a sequence.
