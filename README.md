# seattle_airbnb_prediction
CSE547 Project: Seattle Airbnb Prices Prediction 

## Data Preprocessing

### Listings

We divided the attributes in the listings files into two types, which are categorical features and quantitative features. Categorical attributes such as neighborhoods, bed types and property types were one-hot encoded into features which only have 0 or 1 values. For quantitative attributes, we performed box-cox transformation and then min-max normalization on them to guarantee their values are distributed relatively uniform from 0 to 1.

As for the location information, we kept the direct features -- latitude, longitude, and the neighborhood of each listing. Furthermore, we picked some attractions, such as space needle and pike place market, as landmarks and computed the distance between them and each property. Then we counted how many attractions are near (less than one mile

### Reviews

For each listing ID, we combined all reviews on it together and applied NLP package TextBlob to extract two features subjectivity and sentimental polarity from the combined comments. The polarity reflects averagely how much users like/dislike one accommodation and subjectivity reflects whether the comments are biased or objective.

### Calendar

The calendar files include the available dates of each listing and the corresponding prices. To utilize seasonality information in our prediction, we transformed each date into years, month and weekday features, and one hot encoded the festivals into features. Also, we computed the time interval between the available date and scraped date for each listing and added it as a feature, to count the potential price adjustment made by hosts if the properties fail to be rented out with the time going. 

## Models

We applied Random Forest and XGBoost in making regression analysis of price estimation based on features, and used grid search method in cross-validation to find the optimal hyperparamters

## Time Series Forecasting

We built a LSTM network with ten hidden units on our price data to do time series analysis and try to predict the price based on previous prices information. We set the look back size as 10, which means for each day, we used the prices of its previous ten days to predict its price. 