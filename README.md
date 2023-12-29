# Singapore Flat Resale Price Prediction

Dashboard: [Sigapore_Flat_Dashboard](https://singapore-falt-resale-price-prediction.onrender.com)

The project will benefit both potential buyers and sellers in the Singapore housing market. Buyers can use the application to estimate resale prices and make informed decisions, while sellers can get an idea of their flat's potential market value. Additionally, the project demonstrates the practical application of machine learning in real estate and web development.

Problem Statement:

The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

Process:

1. The train dataset has all of the columns needed to generate and refine the models. The test dataset has all of those columns except for the target variable (resale price).

2. Generate regression model using the training data. This process consists of: Data Cleaning,EDA,Feature Engineering,Train-test split,Models( DecisionTree,Random Forest Regressor,KNeighbors Regressor)

3. Use of train-test split, cross-validation
4. Evaluate models (Model with high R^2 score and low MSE is the optimal model to choose)
