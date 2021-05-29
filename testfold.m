clear all;
pack();
Images = loadMNISTImages('./MNIST/t10k-images.idx3-ubyte_');
Images = reshape(Images, 28,28,[]);
Labels = loadMNISTLabels('./MNIST/t10k-labels.idx1-ubyte_');
Labels(Labels == 0) = 10 ; % 0--> 10

c = cvpartition(Labels,'KFold',5);