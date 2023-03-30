# Towards_Understanding_How_DA_Works
This repository provides python code and links to data sources that support experiments in the paper, "Towards Understanding How Data Augmentation Works when Learning with Imbalanced Data," by Damien Dablain and Nitesh Chawla.

For single layer models (SVM and logistic regression - LG), we used the SKLearn package to train and predict with tabular data. Information about SVM support vectors and LG weights can be conveniently extracted from SKLearn fitted models with built-in functions. We have included sample extracted data in the attached links to reproduce representative experiments.

SV_viz.py can be used to dispaly the following visualizations relating to SVM models: Ratio of Class Dual Coefficient Values, Ratio of Number of Class Support Vectors, Ratio of New Support Vectors vs Base, and the Ratio of Synthetic Support Vectors.

SV_counts.py generates the files contained in SV_viz.py.

The change in model weights for the image datasets can be calculated with cifar_wt_diff.py, places_wt_diff.py, and inat_wt_diff.py for CIFAR-10, Places, and INaturalist, respectively. Pre-trained models are available through the data link.



Data and pre-trained models are available at XX.


