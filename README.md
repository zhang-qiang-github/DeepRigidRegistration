# DeepRigidRegistration
test the deep learning for rigid registration

the folder of data contains 100 images (about 5 M)

Run: python main.py

However, the loss is very large:

iteration: 1476  || loss = 0.102  || error (degree) = 49.990
iteration: 1477  || loss = 0.077  || error (degree) = 42.407
iteration: 1478  || loss = 0.075  || error (degree) = 45.109
iteration: 1479  || loss = 0.015  || error (degree) = 17.671
iteration: 1480  || loss = 0.039  || error (degree) = 31.031
iteration: 1481  || loss = 0.034  || error (degree) = 25.559
iteration: 1482  || loss = 0.086  || error (degree) = 48.926
iteration: 1483  || loss = 0.043  || error (degree) = 32.719
iteration: 1484  || loss = 0.051  || error (degree) = 32.231
iteration: 1485  || loss = 0.101  || error (degree) = 48.668
iteration: 1486  || loss = 0.043  || error (degree) = 31.668
iteration: 1487  || loss = 0.064  || error (degree) = 43.022
iteration: 1488  || loss = 0.071  || error (degree) = 39.949
iteration: 1489  || loss = 0.045  || error (degree) = 36.031
iteration: 1490  || loss = 0.041  || error (degree) = 27.933
iteration: 1491  || loss = 0.061  || error (degree) = 38.986
iteration: 1492  || loss = 0.053  || error (degree) = 38.010
iteration: 1493  || loss = 0.093  || error (degree) = 48.211
iteration: 1494  || loss = 0.039  || error (degree) = 31.405
iteration: 1495  || loss = 0.090  || error (degree) = 46.062
iteration: 1496  || loss = 0.056  || error (degree) = 36.753
iteration: 1497  || loss = 0.046  || error (degree) = 31.771
iteration: 1498  || loss = 0.085  || error (degree) = 45.942
iteration: 1499  || loss = 0.078  || error (degree) = 46.909
iteration: 1500  || loss = 0.043  || error (degree) = 32.242
iteration: 1501  || loss = 0.034  || error (degree) = 26.049
iteration: 1502  || loss = 0.028  || error (degree) = 23.507
iteration: 1503  || loss = 0.052  || error (degree) = 35.817
iteration: 1504  || loss = 0.071  || error (degree) = 45.900
iteration: 1505  || loss = 0.021  || error (degree) = 22.541
iteration: 1506  || loss = 0.041  || error (degree) = 30.613
iteration: 1507  || loss = 0.066  || error (degree) = 38.762
iteration: 1508  || loss = 0.040  || error (degree) = 34.645
iteration: 1509  || loss = 0.032  || error (degree) = 29.264
iteration: 1510  || loss = 0.020  || error (degree) = 18.358

The error is: abs(pred_rotation_angle - true_rotation_angle). The error is very large.
