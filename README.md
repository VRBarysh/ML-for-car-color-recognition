# ML-for-car-color-recognition
Car color regognition model: Single Shot Detector -> Fully convolutional for segmentation -> 2D histogram -> classification

This is my solution of a test task where I had to detect cars and classify them by color on pictures. I've achieved over 90% accuracy over 8 color classes (black, blue, green, grey, orange, red, white, yellow) from the test part of the dataset with almost half of the errors corresponding to wrong or questinable labels. However, my solution has some problems with distinguishing between grey, white and black cars.

I've used the following dataset for training and validation of the classifier: 
https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset

This solution consists of the following steps:
1. Use a ResNet-based pre-trained model of SSD (Single Shot Detector) architecture from Pytorch to localize cars with bounding boxes.
2. Use a ResNet-based pre-trained model of a fully convolutional architecture from Pytorch to create a pixel mask.
3. Use the bounding boxes and the mask to locate pixels that belong to individual cars.
4. Create a 2D histogram from hue and saturation values of those pixels.
5. Create a histogram from brightness values of thos pixels.
6. Use an ensemble of gradient boosting from LightGBM and sklearn's support vector machine models for classification with histograms as features.

![alt text](https://github.com/VRBarysh/ML-for-car-color-recognition/blob/image.jpg?raw=true)
