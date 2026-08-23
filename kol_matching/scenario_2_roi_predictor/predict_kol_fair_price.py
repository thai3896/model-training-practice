import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

print("1. Loading the KOL dataset...")
# Load the data we generated
df = pd.read_csv("kol_data.csv")

# Display the first few rows to understand what we're working with
print("\nDataset Preview:")
print(df[['Name', 'Follower_Count', 'Engagement_Rate', 'Avg_Cost_Per_Post']].head())

print("\n2. Preprocessing Data (Turning text into math)...")
# Machine learning models only understand numbers, not text like "TikTok" or "Beauty"
# So we use a LabelEncoder to convert them (e.g., TikTok = 0, Instagram = 1)
le_platform = LabelEncoder()
le_niche = LabelEncoder()

df['Platform_Code'] = le_platform.fit_transform(df['Primary_Platform'])
df['Niche_Code'] = le_niche.fit_transform(df['Main_Niche'])

# Define our Features (X) and our Target (y)
# We want to predict the Cost based on Platform, Followers, Engagement, and Niche
X = df[['Platform_Code', 'Follower_Count', 'Engagement_Rate', 'Niche_Code']]
y = df['Avg_Cost_Per_Post']

# Split the data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n3. Training the XGBoost Model...")
# XGBoost builds a series of decision trees to learn the patterns in the data
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

print("Training complete!")

print("\n4. Evaluating the Model...")
# Ask the model to predict costs for the 20% of data it didn't see during training
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"On average, the model's predictions are off by: ${error:.2f}")

print("\n5. Testing a brand new, fake KOL!")
# Let's invent a new KOL: A TikToker in the Tech niche with 250,000 followers and an insanely high 0.12 (12%) engagement rate
new_platform = le_platform.transform(['TikTok'])[0]
new_niche = le_niche.transform(['Tech'])[0]

# [Platform_Code, Follower_Count, Engagement_Rate, Niche_Code]
new_kol_data = pd.DataFrame({
    'Platform_Code': [new_platform],
    'Follower_Count': [250000],
    'Engagement_Rate': [0.12],
    'Niche_Code': [new_niche]
})

predicted_cost = model.predict(new_kol_data)[0]
print(f"The model predicts this KOL should charge: ${predicted_cost:.2f} per post.")
