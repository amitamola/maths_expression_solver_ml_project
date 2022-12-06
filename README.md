
# Mathematical Expression Solver
## Machine Learning Project - Group 73 
This repository is part of our Machine Learning Project at Trinity College Dublin
--
**NOTE - Use the requirements.txt file to install the necessary libraries**

## <ins>Dataset</ins> 
The dataset for re-training our models is available at https://drive.google.com/drive/folders/14RE3JyYX-pYkvfboRQ_xTXG99wckY2Iq?usp=share_link

Folder description:
- **data related work** - raw data from data collection from each member along with captioning and bounding box annotation jsons. [JSON to CSV.ipynb](https://github.com/amitamola/maths_expression_solver_ml_project/blob/main/JSON%20to%20CSV.ipynb "JSON to CSV.ipynb") can be used to convert JSONs to CSV format to be used later again.
- **final_symbols_with_split** - data for SVM and Logistic model training
- **final_symbols_split_ttv** - data for CNN model with train, test and validation(ttv) split
-------------

## <ins>Training phase</ins>
We have added the training scripts and files inside "final_codebase/training_phase/" that we used for training all three of our classification models. There are folders for each. The data can be downloaded from the link above in Dataset section.

- [CNN](https://github.com/amitamola/maths_expression_solver_ml_project/blob/main/final_codebase/training_phase/cnn/CNN%20Classifier%20Notebook.ipynb "CNN Classifier Notebook.ipynb")
- [Multinomial Logistic](https://github.com/amitamola/maths_expression_solver_ml_project/blob/main/final_codebase/training_phase/logistic/LR_Math_Expression_Final.ipynb "LR_Math_Expression_Final.ipynb")
- [SVM](https://github.com/amitamola/maths_expression_solver_ml_project/blob/main/final_codebase/training_phase/svm/SVM_Math_Expression_Final.ipynb "SVM_Math_Expression_Final.ipynb")

---
## <ins> Notebooks</ins>
- [Main Notebook.ipynb](https://github.com/amitamola/maths_expression_solver_ml_project/blob/main/Main%20Notebook.ipynb "Main Notebook.ipynb") -To run inference and try the solution on any image
**Note** - *unzip the test_images.zip before running detection on them.*
- [Text localization.ipynb](https://github.com/amitamola/maths_expression_solver_ml_project/blob/main/Text%20localization.ipynb "Text localization.ipynb") - Notebook to see Text Localization code in works
- [JSON to CSV.ipynb](https://github.com/amitamola/maths_expression_solver_ml_project/blob/main/JSON%20to%20CSV.ipynb "JSON to CSV.ipynb") - Notebook useful to convert captioning and annotations file from JSON to CSV format
- [Image_Augmentation.ipynb](https://github.com/amitamola/maths_expression_solver_ml_project/blob/main/Image_Augmentation.ipynb "Image_Augmentation.ipynb") - Notebook used to perform image augmentation which was done as to create synthetic data creation for training purpose.
