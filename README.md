# Code---Enhancing-COVID-19-Fake-News-Detection-with-FastText-Embeddings-and-CNNs

This project focuses on detecting fake news related to COVID-19 using multiple machine learning and deep learning models. Special attention is given to enhancing performance using FastText word embeddings combined with a CNN architecture. A comparison between models using FastText embeddings and models without embeddings is also provided, including results after hyperparameter tuning.
Overview
The COVID-19 pandemic has been accompanied by an "infodemic" of fake news and misinformation. Accurate and efficient fake news detection models are essential to mitigate this impact.
In this project:
•	We leverage both traditional ML classifiers and deep learning models.
•	We compare performance with and without FastText embeddings.
•	We perform hyperparameter tuning to optimize our CNN model.
Dataset
We used the COVID-19 Fake News Infodemic Research Dataset (COVID19-FNIR Dataset), which contains:
•	News articles and social media posts
•	Labels: 0 for fake, 1 for real
Data preprocessing steps include:
•	Text cleaning (removing URLs, special characters)
•	Tokenization
•	Lowercasing
•	Stopword removal
•	Lemmatization
________________________________________
Models Used
We experimented with both traditional ML models and a CNN deep learning model.
Traditional Machine Learning Models:
•	 TF-IDF + Random Forest
•	 TF-IDF + Extra Trees
•	 TF-IDF + Gradient Boosting Machine (GBM)
•	 TF-IDF + Stochastic Gradient Descent (SGD)
Deep Learning Model:
•	 CNN (without embeddings)
•	 CNN + FastText Embeddings (enhanced model)
________________________________________
 Experiments
•	 Models were trained with and without FastText embeddings.
•	 Hyperparameter tuning was applied to the CNN models to optimize:
o	Filter sizes
o	Number of filters
o	Dropout rates
o	Learning rates
o	Optimizers (Adam, RMSProp, SGD)
________________________________________
🧠 CNN + FastText Model Architecture
Input Layer (Preprocessed Text)
        ↓
FastText Embedding Layer (300-d)
        ↓
1D Convolution Layers (multiple filter sizes)
        ↓
Max Pooling
        ↓
Flatten Layer
        ↓
Fully Connected Dense Layers
        ↓
Sigmoid Output (Fake / Real)


 Requirements
Install required packages using:
pip install -r requirements.txt
Main libraries:
•	Python 3.7+
•	numpy
•	pandas
•	scikit-learn
•	tensorflow / keras
•	fasttext
•	matplotlib (for plots)
________________________________________
Usage
1. Preprocess the Dataset
python preprocess.py --input data/raw.csv --output data/cleaned.csv
2. Train ML Models
python train_ml_models.py --dataset data/cleaned.csv
3. Train CNN + FastText
python train_cnn_fasttext.py --embedding_path data/cc.en.300.vec.gz --dataset data/cleaned.csv
4. Evaluate Models
python evaluate.py --model_path models/best_model.h5 --dataset data/test.csv

Hyperparameter Tuning
We applied hyperparameter tuning on CNN using:
•	Grid Search over filter sizes, dropout, learning rate
•	Optimizers: Adam, SGD, RMSProp
•	Batch sizes: 32, 64, 128
Result: Tuned CNN + FastText model significantly outperformed baseline models.
Contributing
Feel free to fork, submit PRs, and suggest improvements!
Open an issue if you encounter bugs or want to propose enhancements.
License
Licensed under the MIT License.

Acknowledgements
•	IEEE DataPort for providing the COVID19-FNIR dataset
•	FastText
•	TensorFlow and Scikit-Learn


