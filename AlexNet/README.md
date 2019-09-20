This is the AlexNet network completed with Pytorch. 
At the beginning, we trained according to the original model.
As follows：
Epoch:0,Train Loss:1.454825,Train Acc:0.465239,Valid Loss:1.220235,Valid Acc:0.556800,Time:28.304s
Epoch:1,Train Loss:1.083338,Train Acc:0.617040,Valid Loss:1.013868,Valid Acc:0.643200,Time:28.119s
Epoch:2,Train Loss:0.929462,Train Acc:0.669520,Valid Loss:0.960434,Valid Acc:0.661000,Time:28.088s
Epoch:3,Train Loss:0.818926,Train Acc:0.711879,Valid Loss:0.902969,Valid Acc:0.679700,Time:28.314s
Epoch:4,Train Loss:0.744624,Train Acc:0.736560,Valid Loss:0.848125,Valid Acc:0.700300,Time:31.140s
Epoch:5,Train Loss:0.676130,Train Acc:0.761480,Valid Loss:0.825201,Valid Acc:0.720200,Time:30.237s
Epoch:6,Train Loss:0.621284,Train Acc:0.781380,Valid Loss:0.878650,Valid Acc:0.705900,Time:28.620s
Epoch:7,Train Loss:0.563401,Train Acc:0.800440,Valid Loss:0.850740,Valid Acc:0.717400,Time:28.474s
Epoch:8,Train Loss:0.513271,Train Acc:0.816880,Valid Loss:0.935455,Valid Acc:0.709900,Time:27.970s
Epoch:9,Train Loss:0.465951,Train Acc:0.834879,Valid Loss:0.899961,Valid Acc:0.723400,Time:26.602s
Epoch:10,Train Loss:0.419050,Train Acc:0.849379,Valid Loss:0.951434,Valid Acc:0.722000,Time:26.865s
Epoch:11,Train Loss:0.379573,Train Acc:0.865519,Valid Loss:1.013515,Valid Acc:0.715000,Time:27.462s
Epoch:12,Train Loss:0.341161,Train Acc:0.876899,Valid Loss:1.118112,Valid Acc:0.712900,Time:32.365s
Epoch:13,Train Loss:0.311653,Train Acc:0.887059,Valid Loss:1.133076,Valid Acc:0.715300,Time:29.395s
Epoch:14,Train Loss:0.281603,Train Acc:0.898799,Valid Loss:1.222735,Valid Acc:0.712800,Time:29.217s
Epoch:15,Train Loss:0.253446,Train Acc:0.908279,Valid Loss:1.223001,Valid Acc:0.707600,Time:33.042s
Epoch:16,Train Loss:0.238096,Train Acc:0.914019,Valid Loss:1.302816,Valid Acc:0.716100,Time:38.240s
Epoch:17,Train Loss:0.215534,Train Acc:0.922040,Valid Loss:1.373315,Valid Acc:0.709700,Time:37.452s
Epoch:18,Train Loss:0.206995,Train Acc:0.924499,Valid Loss:1.438006,Valid Acc:0.704800,Time:29.068s
Epoch:19,Train Loss:0.191343,Train Acc:0.930899,Valid Loss:1.593027,Valid Acc:0.701100,Time:30.737s

We can see that at the end of the training, the model began to overfitting, and later tried to join the Dropout layer.
We can see the result is:
Epoch:0,Train Loss:1.535922,Train Acc:0.430700,Valid Loss:1.272498,Valid Acc:0.525100,Time:38.420s
Epoch:1,Train Loss:1.212233,Train Acc:0.565340,Valid Loss:1.091377,Valid Acc:0.606400,Time:40.391s
Epoch:2,Train Loss:1.071773,Train Acc:0.620940,Valid Loss:1.022614,Valid Acc:0.638300,Time:36.469s
Epoch:3,Train Loss:0.982558,Train Acc:0.652979,Valid Loss:0.929434,Valid Acc:0.675900,Time:30.581s
Epoch:4,Train Loss:0.913655,Train Acc:0.677719,Valid Loss:0.862355,Valid Acc:0.705300,Time:28.231s
Epoch:5,Train Loss:0.856286,Train Acc:0.699419,Valid Loss:0.834224,Valid Acc:0.710500,Time:38.862s
Epoch:6,Train Loss:0.818004,Train Acc:0.710939,Valid Loss:0.799539,Valid Acc:0.723800,Time:47.371s
Epoch:7,Train Loss:0.779099,Train Acc:0.725120,Valid Loss:0.823127,Valid Acc:0.717200,Time:38.584s
Epoch:8,Train Loss:0.747750,Train Acc:0.737600,Valid Loss:0.766669,Valid Acc:0.736400,Time:35.819s
Epoch:9,Train Loss:0.721264,Train Acc:0.743579,Valid Loss:0.790680,Valid Acc:0.730900,Time:27.828s
Epoch:10,Train Loss:0.702976,Train Acc:0.751140,Valid Loss:0.753237,Valid Acc:0.744600,Time:33.311s
Epoch:11,Train Loss:0.675494,Train Acc:0.763020,Valid Loss:0.773966,Valid Acc:0.739600,Time:37.997s
Epoch:12,Train Loss:0.652072,Train Acc:0.768200,Valid Loss:0.772406,Valid Acc:0.739400,Time:37.417s
Epoch:13,Train Loss:0.637377,Train Acc:0.774860,Valid Loss:0.760263,Valid Acc:0.743300,Time:36.886s
Epoch:14,Train Loss:0.617265,Train Acc:0.780540,Valid Loss:0.755194,Valid Acc:0.749200,Time:37.993s
Epoch:15,Train Loss:0.604778,Train Acc:0.783900,Valid Loss:0.766801,Valid Acc:0.743600,Time:36.724s
Epoch:16,Train Loss:0.591209,Train Acc:0.789221,Valid Loss:0.770330,Valid Acc:0.740900,Time:42.752s
Epoch:17,Train Loss:0.580997,Train Acc:0.792660,Valid Loss:0.776271,Valid Acc:0.743700,Time:37.608s
Epoch:18,Train Loss:0.563570,Train Acc:0.798739,Valid Loss:0.778380,Valid Acc:0.741100,Time:35.571s
Epoch:19,Train Loss:0.556388,Train Acc:0.803520,Valid Loss:0.786071,Valid Acc:0.738300,Time:35.016s


We can see that the training effect of the model has improved by four. percentage point.
