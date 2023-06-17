# Resume-Predictor
The provided Python script is part of a Resume Analyzer. The tool allows users to input resumes and, using Natural Language Processing techniques, it extracts information from the resumes, visualizes the extracted data, and classifies resumes into different job categories using machine learning algorithms.

## Key Features
1. **Data Preprocessing:** The program preprocesses resume data to remove any noise that could potentially affect the performance of the NLP model. The preprocessing tasks include text normalization, stopword removal, and text tokenization and stemming.
2. **Data Visualization:** Once the data is preprocessed, the script performs several visualizations to better understand the underlying patterns. It generates plots for category distribution, word frequency, skill distribution, and the most used words per job category. It also creates an entity recognition plot and a dependency tree plot using the DisplaCy visualizer from SpaCy.
3. **Data Processing with SpaCy:** The script applies SpaCy to perform NLP tasks. It defines entities for skill extraction from resumes, cleans resume data, and plots visualizations based on the processed data.
4. **Model Training:** The script implements a generic model trainer class that can be used to train any Scikit-Learn model. As a demonstration, it trains an XGBoost classifier. The training process involves splitting the data, vectorizing the data, tuning the hyperparameters of the model using GridSearchCV, making predictions, and evaluating the model.

## Motivation
The aim of this script is to provide an automated way to analyze and classify resumes. This can be particularly useful in recruitment scenarios, where it can help to quickly sort through large numbers of resumes and identify the ones that are most relevant to specific job categories. By applying NLP and machine learning, the script extracts meaningful insights from unstructured text data (i.e., the resumes) and performs tasks that would be time-consuming and error-prone if done manually.

## Setup Environment
```bash
git clone https://github.com/davidlevinwork/Resume-Predictor.git
cd GB-AFS
pip install -r requirements.txt
python main.py
```
Please notice that:
* **Dataset:** The script expects a CSV file named Resume.csv inside a folder named Resume at the root directory. Make sure the dataset is available in the correct path.
* **Output Folders:** The script saves the outputs to several directories. Make sure the following directories are available in your root directory: Outputs, Outputs/Models, Outputs/Plots, and Outputs/log.log.
* **Skill Patterns:** The script expects a file named jz_skill_patterns.jsonl for entity extraction using SpaCy. Ensure this file is in your root directory.
