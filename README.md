# ACTS
Automated XCT segmentation https://doi.org/10.1016/j.addma.2020.101460
This is code relevant for the ACTS tool (Automated Computed Tomography Segmentation) for X-ray computed tomogrpay data sets. 
The link to the paper I, and gratefully others, have published is provided. The paper details the methodology for building, training and implementing the ACTS tool. 
The code provided in this repository is free to use under condition that you cite the given article. 
The code provided allows for OTSU tranform of image stacks, training your own ACTS network on a data set, and using ACTS to segment other data.  A pre-trained ACTS neural network can be provided if asked. 

** Importance on file structure

OTSU_transforms takes in 16-bit .tif files and coverts them to 16-bit .png files
*raw images and label files should have the exact same file name if training
"data_set_folder" variable should be the same across each script. 
The data_set_folder be a directory which contains the following labled directories, the first should be created by the user:
  
  -"Raw" - contains directories , each titled after a specimen name, which contain the .tif images from the XCT scan
  -"Labels" -contains directories , each titled after a specimen name, which contain 8-bit .png with max value as foreground and min value as background/pore
  -"Images" - contain directories, each titled after a specimen name, and created from running the OTSU_transform, each specimen directory in the "Images" folder contains 16-bit .png images
  -"Predictions" - contain directories, each titled after a specimen name, and created from running ACTS_predict.py, each specimen directory in the "Predictions" folder contains 8-bit .png images of ACTS predictions of those images in the "Images folder"
  -"Training_results" created from the ACTS_train.py script, this folder contains the training results, validation stats, and network weights/models generated from the ACTS_train.py script
  


