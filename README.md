**📝 Fake Currency Detection System using Machine Learning**
**📌 Overview**
This project is an AI-powered system that detects fake currency notes using image processing and machine learning techniques.
The system extracts visual features from currency images and uses a trained Decision Tree classifier to determine whether a note is Real or Fake.
A Flask-based web interface allows users to upload a note image and get instant predictions.
________________________________________
**🚀 Features**
•	Automatic detection of Real or Fake currency notes
•	Image preprocessing and feature extraction using OpenCV
•	Machine Learning models:
o	Decision Tree (final and best-performing model)
o	SVM
o	Logistic Regression (used to test linear separability)
•	Flask web application for real-time prediction
•	ROC Curve and Confusion Matrix visualization
•	Supports 60:40, 70:30, 80:20, and 90:10 train-test split evaluations
________________________________________
**📂 Project Structure**
|-- dataset/
|   |-- real/
|   |-- fake/
|
|-- static/
|   |-- css/
|   |-- uploads/
|
|-- templates/
|   |-- index.html
|
|-- app.py
|-- decision_tree_model.pkl
|-- README.md
________________________________________
**🛠 Technologies Used**
Programming & Frameworks
•	Python
•	Flask (Backend Web Framework)
Libraries
•	OpenCV
•	NumPy
•	Pandas
•	Scikit-learn
•	Matplotlib
•	Seaborn
•	Joblib
________________________________________
**🧠 Machine Learning Workflow**
1. Image Preprocessing
•	Resize all images to 128×128
•	Convert to Grayscale and HSV
•	Extract features:
o	Mean & Standard Deviation (BGR, HSV)
o	Laplacian variance (texture/sharpness)
2. Model Training
Models trained and compared:
•	Decision Tree (best performance)
•	SVM
•	Logistic Regression
Decision Tree achieved the highest accuracy (≈99%).
3. Performance Evaluation
Evaluated using:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	ROC Curve & AUC
•	Confusion Matrix
Tested under multiple split ratios:
•	60:40
•	70:30
•	80:20
•	90:10
________________________________________
**💻 Running the Project**
1. Clone the Repository
git clone https://github.com/yourusername/fake-currency-detection.git
cd fake-currency-detection
2. Install Required Libraries
pip install -r requirements.txt
3. Run the Flask App
python app.py
Visit:
http://127.0.0.1:5000
________________________________________
**🖥 Web Interface**
•	Upload a currency note image
•	System extracts features
•	Displays output:
<img width="1110" height="858" alt="image" src="https://github.com/user-attachments/assets/853c0959-0ab2-42e5-a27a-ef723951ef0d" />

o	“This is a Real Currency”
<img width="1309" height="853" alt="image" src="https://github.com/user-attachments/assets/e740ffe5-dabd-49d5-8f18-cfef5c935694" />

o	“This is a Fake Currency”
________________________________________
**📌 Conclusion**
This project demonstrates the effectiveness of machine learning in detecting counterfeit currency notes.
Through detailed evaluation, the Decision Tree classifier proved to be the most accurate and reliable model for deployment.
________________________________________
**🤝 Contributions**
Feel free to open issues or submit pull requests for improvements!


