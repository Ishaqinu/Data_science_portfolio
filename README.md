# Data_science_portfolio
_Welcome to my data science portpolio_

This repository containing portfolio of data science projects completed by me for academic, self learning, and hobby purposes. Presented in the form of Jupyter notebook
## Contents
### Machine Learning
[Road Accident Serverity](https://github.com/Ishaqinu/Data_science_portfolio/blob/main/Road%20accidents%20severity.ipynb) : The data set has has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms .
model used:
* KNN
* NaiveBayes
* SVM
* Decisiontree
* RandomForest

[Car Details](https://github.com/Ishaqinu/Data_science_portfolio/blob/main/car%20details.ipynb) : IN this model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the Linear regression model will be a good way for management to understand the pricing dynamics of a new market.

[Laptop Price Prediction](https://github.com/Ishaqinu/Data_science_portfolio/blob/main/laptop%20price%20prediction.ipynb) : This notebook is about the Price Prediction of a dataset contaning data about laptops. We'll use pandas, numpy and other python packages for data manipulation, matplotlib and seaborn for data visualization & Regression model are used make ml models.
model used:
* linear regression
* KNeighbor
* DecisionTree
* RandomForest
* SVM

[Stock Price Predicition](https://github.com/Ishaqinu/Data_science_portfolio/blob/main/project%20nestle.ipynb) : Before Investing into any company an investor should study Historical stock Prices of that company , analyze the Opening-closing and High/Low dPrices for better understanding of the performance of that company in share markets. Here we are predicting the Closing Price of Nestle Shares with the available Independent Features.
From this Data we can get to know :
* Which Feature or Features are significant in predicting the close Price.
* How well those Features describe the Close price of this Company.

[Weather dataset ](https://github.com/Ishaqinu/Data_science_portfolio/commit/2a0bc1ad6c1c3f18aba40e6329c3d5346d9be8d9): This a comprehensive midsized Dataset on Weather records in different areas of Australia. This can be used for beginner level ML binary Classification models to predict whether it will rain or not Tomorrow . i have used encoder and imputer iinthis dataset

### NLP
[Emotion Dataset](https://github.com/Ishaqinu/Data_science_portfolio/blob/main/Emotions.ipynb) : This project is an example of how NLP can be used to classify emotions in text. This type of technology can be used to develop a variety of applications, such as sentiment analysis and customer feedback analysis.
Here are some additional details about the steps involved in this project:
* using spacy
* Tfidf vecotization
* model: RandomForest (classification)

[IMDB rating](https://github.com/Ishaqinu/Data_science_portfolio/blob/main/IMDB%20rating.ipynb) : this dataset contain 25000 review of movies . i have predicted
weather the review is positive or negative using NLP
Here are some additional details about the steps involved in this project:
* count vectorizer
* snowball stemmer
* model: Navie Bayes

### Deep Learning
[Face Recognition-based Attendance System with Absent Student Notification](https://github.com/Ishaqinu/Data_science_portfolio/blob/main/attendence%20project/message.py) : this project aims to automate attendance tracking by using face recognition technology. It captures video from a webcam, detects faces in real-time, compares them with a set of known faces, and marks the attendance of recognized individuals afterthat machine will notify individuals sending sms to them
- Dependencies: The project relies on several Python libraries -
  * OpenCV for computer vision tasks
  * numpy for numerical operations
  * face_recognition for face detection and recognition
  * os for file and directory operations
  * twilio for sending SMS notifications
  * pandas for data manipulation
 
[Hand Gesture Drawing Application](https://github.com/Ishaqinu/Data_science_portfolio/blob/main/Hand%20Gesture%20Drawing/main.py) : The Hand Gesture Drawing Application is a computer vision project that allows users to draw on a digital canvas using their hand gestures. The application utilizes real-time hand detection and tracking techniques to interpret the user's hand movements and translate them into drawing actions on the screen
 - Dependencies: The project relies on several Python libraries -
  * OpenCV for computer vision tasks
  * mediapipe utilized for hand tracking and landmark detection.
  * numpy for numerical operation
  * hand_recognition for hand detection and recognition
 -Hand Tracking: The project utilizes the MediaPipe library to track hand movements and landmarks in real-time from the webcam feed. The handDetector class is responsible for managing hand tracking operations.
 -Drawing: Users can draw on the canvas by making specific hand gestures. When the thumb and two fingers are extended, drawing is enabled. Different colors can be selected by positioning the hand in specific 
    regions of the frame (color rectangles)
