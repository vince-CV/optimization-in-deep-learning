# Convex-Optimizer-in-Deep-Learning


 Authors notice:

 DL got developed thanks to the a variaty of techniques such as layer, gradient update, initialization, non-linear, and normalization, but optimization in DL are usually non-convex problem. 

 GD: using full batch of dataset, and easier to stuck into local minima (cost functions in DL are always not that flat). Also large computation requires;

 SGD: using single data, and this kind of strategy introduced larger noise so that the training can jump out from saddle points. This method tends to be slow, and performances will have dramatical vibrataion.

 MiniBatch GD: using a certain part of dataset. Usually it's not intuitive to find a optimal value for batch size. Too large Batch size limited by GPU storge, but too small Batch can also limited by time.

 1st Order: SGD, Adagrad, Adam..., Batch size would be around hundreds;
2nd Order: Conj-gradient, Newton, Quasi-Newton, L-BFGS... the errors would be skyrocketing if the 1st order have obviously errors; Even though the smaller batch can speed-up convergence, it also introduced much noise which will adversly affects the performance. So batch size would be thousands or more for 2nd order optimization methods;

 In the other hand, smaller batch are not that stable, so I always increased the learning rate to fix this gap, otherwise the model could not convergent since the malignant oscillation.

 Experience from my DL book:
1. slightly increase batch size step-by-step;
2. using SGD near the end of training;
3. small batch to introduce noise at the earlier stage, and large batch to get rid of oscillation later.
