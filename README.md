# Mars-surface-image-segmentation-dataset-for-Tianwen-1-Mission  
  
The files in **make_dataset** can help you convert json files to label images and generate .txt documents required for training  
  
There are six subfolders **train, val, test, trainannot, valannot, testannot under the data folder**, and the corresponding original images and label images are placed for network training.  
  
There are 5 .py files placed in the **network** folder, which are the codes of the 5 combined networks. Just modify the path where the dataset is placed in the .py file to run.  
  
The experimental environment requirements are specified in the **requirement.txt** document. 
