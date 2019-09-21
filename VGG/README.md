Implementation of VGG based on Pytorch.

Initially, based on the original model, the training results are as follows:

Epoch:0,Train Loss:1.800689,Train Acc:0.305120,Valid Loss:1.502848,Valid Acc:0.425800,Time:113.860s
Epoch:1,Train Loss:1.305084,Train Acc:0.520800,Valid Loss:1.207958,Valid Acc:0.570000,Time:110.802s
Epoch:2,Train Loss:1.060692,Train Acc:0.620460,Valid Loss:1.048332,Valid Acc:0.634500,Time:113.817s
Epoch:3,Train Loss:0.905357,Train Acc:0.677800,Valid Loss:0.960315,Valid Acc:0.669600,Time:125.922s
Epoch:4,Train Loss:0.786200,Train Acc:0.721220,Valid Loss:0.869613,Valid Acc:0.704100,Time:129.467s
Epoch:5,Train Loss:0.693650,Train Acc:0.758840,Valid Loss:0.947766,Valid Acc:0.683100,Time:141.611s
Epoch:6,Train Loss:0.615341,Train Acc:0.785340,Valid Loss:0.850214,Valid Acc:0.707800,Time:139.237s
Epoch:7,Train Loss:0.536505,Train Acc:0.813080,Valid Loss:0.847145,Valid Acc:0.729200,Time:148.003s
Epoch:8,Train Loss:0.472259,Train Acc:0.835719,Valid Loss:0.854124,Valid Acc:0.726300,Time:152.757s
Epoch:9,Train Loss:0.406855,Train Acc:0.858959,Valid Loss:0.872565,Valid Acc:0.732500,Time:151.320s
Epoch:10,Train Loss:0.353191,Train Acc:0.878139,Valid Loss:0.933212,Valid Acc:0.721000,Time:141.744s
Epoch:11,Train Loss:0.319918,Train Acc:0.889719,Valid Loss:0.979963,Valid Acc:0.729000,Time:141.313s
Epoch:12,Train Loss:0.283307,Train Acc:0.901639,Valid Loss:0.986029,Valid Acc:0.734200,Time:145.218s
Epoch:13,Train Loss:0.238209,Train Acc:0.918220,Valid Loss:1.159613,Valid Acc:0.735400,Time:159.451s
Epoch:14,Train Loss:0.210888,Train Acc:0.928020,Valid Loss:1.315534,Valid Acc:0.726700,Time:155.328s
Epoch:15,Train Loss:0.196895,Train Acc:0.932140,Valid Loss:1.170377,Valid Acc:0.729400,Time:150.371s
Epoch:16,Train Loss:0.179566,Train Acc:0.939920,Valid Loss:1.273586,Valid Acc:0.719800,Time:140.518s
Epoch:17,Train Loss:0.167506,Train Acc:0.944740,Valid Loss:1.288458,Valid Acc:0.725300,Time:150.624s
Epoch:18,Train Loss:0.157826,Train Acc:0.948320,Valid Loss:1.398109,Valid Acc:0.729200,Time:155.814s
Epoch:19,Train Loss:0.145865,Train Acc:0.952859,Valid Loss:1.341556,Valid Acc:0.724600,Time:156.283s

It can be seen that VGG can achieve 72% accuracy on the CIFAR10 dataset, but it is not difficult to find that the model has been over-fitting. After adding BatchNorm2d and Dropout to the model, the effect is as follows:

Epoch:0,Train Loss:1.462059,Train Acc:0.443540,Valid Loss:1.445575,Valid Acc:0.499100,Time:128.083s
Epoch:1,Train Loss:1.001163,Train Acc:0.641920,Valid Loss:1.038049,Valid Acc:0.639000,Time:147.030s
Epoch:2,Train Loss:0.812884,Train Acc:0.714680,Valid Loss:0.859768,Valid Acc:0.708400,Time:161.040s
Epoch:3,Train Loss:0.674642,Train Acc:0.765400,Valid Loss:0.801056,Valid Acc:0.730800,Time:182.816s
Epoch:4,Train Loss:0.581272,Train Acc:0.801020,Valid Loss:0.822073,Valid Acc:0.758400,Time:187.997s
Epoch:5,Train Loss:0.505057,Train Acc:0.826819,Valid Loss:0.771452,Valid Acc:0.774200,Time:176.849s
Epoch:6,Train Loss:0.443718,Train Acc:0.848119,Valid Loss:0.684441,Valid Acc:0.788400,Time:176.942s
Epoch:7,Train Loss:0.387575,Train Acc:0.868499,Valid Loss:0.684636,Valid Acc:0.796000,Time:181.327s
Epoch:8,Train Loss:0.348998,Train Acc:0.878659,Valid Loss:0.909229,Valid Acc:0.787600,Time:182.587s
Epoch:9,Train Loss:0.315689,Train Acc:0.893039,Valid Loss:1.201652,Valid Acc:0.790300,Time:185.507s
Epoch:10,Train Loss:0.278663,Train Acc:0.904419,Valid Loss:0.957212,Valid Acc:0.802600,Time:186.441s
Epoch:11,Train Loss:0.244609,Train Acc:0.915280,Valid Loss:0.782118,Valid Acc:0.803400,Time:191.963s
Epoch:12,Train Loss:0.224417,Train Acc:0.923780,Valid Loss:0.688443,Valid Acc:0.805900,Time:195.127s
Epoch:13,Train Loss:0.200347,Train Acc:0.931419,Valid Loss:0.931186,Valid Acc:0.796100,Time:195.170s
Epoch:14,Train Loss:0.181293,Train Acc:0.936219,Valid Loss:0.843933,Valid Acc:0.809700,Time:199.928s
Epoch:15,Train Loss:0.164428,Train Acc:0.943480,Valid Loss:0.870400,Valid Acc:0.805900,Time:194.434s
Epoch:16,Train Loss:0.162795,Train Acc:0.943660,Valid Loss:0.763400,Valid Acc:0.808000,Time:189.437s
Epoch:17,Train Loss:0.145443,Train Acc:0.950260,Valid Loss:0.854368,Valid Acc:0.811800,Time:192.165s
Epoch:18,Train Loss:0.131465,Train Acc:0.955180,Valid Loss:1.561488,Valid Acc:0.820300,Time:218.043s
Epoch:19,Train Loss:0.130171,Train Acc:0.955719,Valid Loss:0.828362,Valid Acc:0.793300,Time:202.336s

It can be found that the same model can also have such a big improvement, and deep learning is really amazing.
