## Description

I implement light models with Pytorch, models are SqueezeNet, ShuffleNet, MobileNet, MobileNetv2 and [ShuffleNetv2](https://www.jiqizhixin.com/articles/2018-07-29-3).

You can get details about these models at [纵览轻量化卷积神经网络：SqueezeNet、MobileNet、ShuffleNet、Xception](https://www.jiqizhixin.com/articles/2018-01-08-6)

you can train the model with the command:

python main.py --model SuqeezeNet --epoch 100 --batch_size 64 --learning_rate 0.03 --use_cuda True


Limited by the computing power, I just verificate the correct of these models on CIFAR-10, don't get the best accuracy.

## References

[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

[Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)

Some parts of [kuangliu's code](https://github.com/kuangliu/pytorch-cifar)

Some parts of [togheppi's code](https://github.com/togheppi/CycleGAN)
