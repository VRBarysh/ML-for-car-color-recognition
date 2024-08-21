# ML-for-car-color-recognition
Car color regognition model: Single Shot Detector -> Fully convolutional for segmentation -> 2D histogram -> classification

This is my solution of a test task where I had to detect cars and classify them by color on pictures. I've achieved over 90% accuracy over 8 color classes (black, blue, green, grey, orange, red, white, yellow) from the test part of the dataset with almost half of the errors corresponding to wrong or questionable labels. However, my solution has some problems with distinguishing between grey, white and black cars.

I've used the following dataset for training and validation of the classifier: 
https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset

This solution consists of the following steps:
1. Use a ResNet-based pre-trained model of [SSD](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD) (Single Shot Detector) architecture to localize cars with bounding boxes.
3. Use a ResNet-based pre-trained model of a fully convolutional architecture to create a pixel mask.
4. Use the bounding boxes and the mask to locate pixels that belong to individual cars.
5. Create a 2D histogram from hue and saturation values of those pixels.
6. Create a histogram from brightness values of thos pixels.
7. Use an ensemble of gradient boosting from LightGBM and sklearn's support vector machine models for classification with histograms as features.

Here are some examples of the resuls:

![Example 01](https://github.com/VRBarysh/ML-for-car-color-recognition/blob/main/examples/image01.jpg?raw=true)
![Example 02](https://github.com/VRBarysh/ML-for-car-color-recognition/blob/main/examples/image02.jpg?raw=true)
![Example 03](https://github.com/VRBarysh/ML-for-car-color-recognition/blob/main/examples/image03.jpg?raw=true)
![Example 04](https://github.com/VRBarysh/ML-for-car-color-recognition/blob/main/examples/image04.jpg?raw=true)
![Example 05](https://github.com/VRBarysh/ML-for-car-color-recognition/blob/main/examples/image05.jpg?raw=true)
![Example 06](https://github.com/VRBarysh/ML-for-car-color-recognition/blob/main/examples/image06.jpg?raw=true)
![Example 07](https://github.com/VRBarysh/ML-for-car-color-recognition/blob/main/examples/image07.jpg?raw=true)
![Example 08](https://github.com/VRBarysh/ML-for-car-color-recognition/blob/main/examples/image08.jpg?raw=true)
![Example 09](https://github.com/VRBarysh/ML-for-car-color-recognition/blob/main/examples/image09.jpg?raw=true)
![Example 10](https://github.com/VRBarysh/ML-for-car-color-recognition/blob/main/examples/image10.jpg?raw=true)
