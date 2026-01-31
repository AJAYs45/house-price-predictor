Headline:  Building an End-to-End Machine Learning Project: From Raw Data to an Interactive Web App!

I am excited to share my latest project, a California House Price Prediction System! 

The goal was not just to train a model, but to build a complete pipeline that handles real-world data challenges and deploy it with a professional UI.

The Technical Journey:

🔹 Data Strategy: instead of a random split, I used StratifiedShuffleSplit on income categories to ensure my training and test sets perfectly represent the population diversity. 
🔹 Pipeline Architecture: Built a robust Scikit-Learn Pipeline using SimpleImputer for missing values, OneHotEncoder for categorical features, and StandardScaler for feature scaling. 
🔹 Model Selection: Experimented with Linear Regression and Decision Trees, but Random Forest Regressor gave the best results with the lowest RMSE. 
🔹 Deployment & UI: Wrapped the model in a Streamlit dashboard. I added a custom "Sci-Fi" style intro with a typewriter effect and Lottie animations to enhance the user experience.
