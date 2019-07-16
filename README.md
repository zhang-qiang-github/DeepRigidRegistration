# DeepRigidRegistration
test the deep learning for rigid registration

the folder of data contains 100 images (about 5 M)

Run: python main.py

When debug this network, I find the BN layer do not work.
For example, I fetch the s1 and s2 (line 82 and 83) as the following images:

![image](https://github.com/zhang-qiang-github/DeepRigidRegistration/blob/master/figure/BN_error.png)

We can find that the value in rs2 is still very large. It looks like that the BN layer do not work. 
