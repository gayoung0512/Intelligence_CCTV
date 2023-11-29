
Intelligence CCTV - v10 augmentation_1
==============================

This dataset was exported via roboflow.com on November 27, 2023 at 7:04 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2073 images.
Objects are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Resize to 640x640 (Fit within)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 70 percent of the image
* Random rotation of between -30 and +30 degrees
* Random shear of between -20° to +20° horizontally and -20° to +20° vertically
* Random brigthness adjustment of between -30 and +30 percent
* Random exposure adjustment of between -20 and +20 percent
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 5 percent of pixels


