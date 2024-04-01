# DeepRelation 

Finding blood relationships using face images can be achieved through deep learning techniques. There are several publicly available datasets, such as Family In the Wild, UTKFace, VGGFace2, and LabelFace, which contain large amounts of face images and their corresponding family relationships. These datasets can be used to train deep learning models to recognize patterns in facial features and predict the relationship between two individuals. Despite the challenges posed by the variability in facial features and the limited training data, the use of advanced techniques such as transfer learning and data augmentation can improve the model's performance.

Deep Relation Project Overview
This project, titled "Deep Relation," explores the potential of deep learning techniques to identify blood relationships between individuals based on facial images.

Objective:

Leverage deep learning models to predict familial connections from facial features.
Train models using publicly available datasets like Family-in-the-Wild, UTKFace, VGGFace2, and LabelFace.
Challenges:

Facial features can vary significantly between individuals, even within families.
The amount of training data available for familial relationships might be limited.
Technologies Used:

Deep learning models (ResNet50)
Transfer learning (potentially)
Data augmentation (potentially)
Implementation Highlights:

The project utilizes the "Kinface-W" dataset for comparing two images and predicting potential relations based on their features.
Two approaches were explored:
Pre-trained ResNet50 model: Achieved an accuracy of 80.86% with a corresponding error rate of 19.14%.

