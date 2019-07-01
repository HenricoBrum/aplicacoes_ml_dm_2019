cluster  1   ['MCHP', 'CINF', 'MDT', 'TMK', 'GPN', 'TEL', 'RCL', 'RJF', 'SWKS', 'LOW']
cluster  2   ['FBHS', 'AJG', 'FAST', 'ROST', 'IP', 'LEN', 'PFG', 'VZ', 'SO', 'NCLH']
cluster  3   ['ESS', 'BIIB', 'CMG', 'BLK', 'SHW', 'CHTR', 'NOC', 'MMM', 'BDX', 'REGN']
cluster  4   ['CHK', 'DRE', 'MGM', 'IPG', 'JNPR', 'PHM', 'AES', 'COTY', 'KEY', 'GE']
cluster  5   ['HON', 'AMP', 'UNP', 'CAT', 'ACN', 'NSC', 'PNC', 'SYK', 'VRTX', 'ITW']
cluster  6   ['FTI', 'LB', 'STX', 'HAL', 'CF', 'VIAB', 'RRC', 'BEN', 'NBL', 'EBAY']
cluster  7   ['DAL', 'ZION', 'BLL', 'HOLX', 'ATVI', 'AMAT', 'DHI', 'CMCSA', 'PGR', 'MAS']
cluster  8   ['MAA', 'MKC', 'DTE', 'EXPE', 'DHR', 'EFX', 'AON', 'VAR', 'ADP', 'JBHT']
cluster  9   ['SPG', 'ROK', 'LH', 'COST', 'PX', 'AYI', 'AMG', 'WAT', 'MTB', 'CMI']
cluster  10   ['SLB', 'COF', 'YUM', 'DOV', 'PG', 'EOG', 'EQT', 'EMN', 'WYN', 'SNI']
cluster  11   ['HIG', 'PEG', 'BK', 'MDLZ', 'SEE', 'NLSN', 'TXT', 'JCI', 'AIV', 'CHD']
cluster  12   ['UTX', 'WYNN', 'PPG', 'AAP', 'RL', 'KMB', 'KSU', 'CVX', 'SRCL', 'SRE']
cluster  13   ['ADBE', 'UNH', 'NFLX', 'LRCX', 'HD', 'ANTM', 'SPGI', 'ALGN', 'HII', 'CTAS']
cluster  14   ['AIG', 'EIX', 'GIS', 'VFC', 'PCG', 'SCG', 'AKAM', 'PCAR', 'BMY', 'EMR']
cluster  15   ['ESRX', 'BAX', 'ETN', 'PNR', 'CL', 'TSCO', 'QCOM', 'TGT', 'TJX', 'VTR']
cluster  16   ['IVZ', 'T', 'GM', 'CAG', 'PPL', 'PWR', 'CA', 'WY', 'FE', 'FOXA']
cluster  17   ['NTRS', 'STT', 'ED', 'MSI', 'TWX', 'PNW', 'D', 'FMC', 'LLY', 'IR']
cluster  18   ['CTL', 'NFX', 'HCP', 'UAA', 'MRO', 'LUK', 'MAT', 'FCX', 'NI', 'GPS']
reading files...
(220846, 31)
(220846,)
evaluating cluster  0
(9824, 30)
fit predict SVR...
fit predict MLP...
SVR MSE:  161.4390512001394
SVR POCID:  100.0
SVR EXACT:  0
SVR THEIL'S U:  12.451773467720617
MLP MSE:  1.7741515758455855
MLP POCID:  100.0
MLP EXACT:  0
MLP THEIL'S U:  1.294058285818514
BASELINE MSE:  1.0580162228990226
BASELINE EXACT:  14
78.58   78.57440842940092   78.77969007892207   78.23
47.66   49.627170951618524   49.513421508849845   48.48
60.0   58.00672878257986   58.11903596956305   58.05
57.5   59.42929114205809   57.487848540307354   56.71
97.29   71.83354177283572   98.39465543443355   97.38
evaluating cluster  1
(9824, 30)
fit predict SVR...
fit predict MLP...
SVR MSE:  10.537074098405055
SVR POCID:  100.0
SVR EXACT:  0
SVR THEIL'S U:  4.620504014682535
MLP MSE:  0.5634226321663915
MLP POCID:  100.0
MLP EXACT:  0
MLP THEIL'S U:  1.129004617790049
BASELINE MSE:  0.44458480317996735
BASELINE EXACT:  29
59.63   54.875031815931635   58.46361836624133   58.27
42.85   42.30191833888709   42.33308426183696   42.2
53.52   52.7679364953808   53.02204045876774   52.52
48.84   48.45378236553223   48.03669703549111   47.7
51.13   51.306837249229105   51.16832372970433   51.0
evaluating cluster  2
(9824, 30)
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
