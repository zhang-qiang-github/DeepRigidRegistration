# DeepRigidRegistration
test the deep learning for rigid registration

the folder of data contains 100 images (about 5 M)

Run: python main.py

I use the CNN to predict the rotation angle. However, the loss is very large:

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


The error is: abs(pred_rotation_angle - true_rotation_angle). The error is very large.
