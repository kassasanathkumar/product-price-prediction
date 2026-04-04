import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import yfinance as yf
import matplotlib.pyplot as plt

# --- Load ML Model & Label Encoders ---
xgb_model = joblib.load("price_predictor.pkl")  # Trained XGBoost model
label_encoders = joblib.load("label_encoders.pkl")  # Dictionary of LabelEncoders

# --- Function to preprocess data ---
def prepare_features(row):
    try:
        name_enc = label_encoders["name"].transform([row["Product Title"]])[0]
    except:
        name_enc = 0  
    try:
        main_cat = label_encoders["main_category"].transform([row.get("Main Category", "Unknown")])[0]
    except:
        main_cat = 0
    try:
        sub_cat = label_encoders["sub_category"].transform([row.get("Sub Category", "Unknown")])[0]
    except:
        sub_cat = 0

    rating = float(row["Rating"]) if row["Rating"] != "N/A" else 0
    reviews = float(row["Reviews"]) if row["Reviews"] != "N/A" else 0

    try:
        actual_price = float(str(row["Price (INR)"]).replace("₹", "").replace(",", "").strip())
    except:
        actual_price = 0

    return np.array([[name_enc, main_cat, sub_cat, rating, reviews, actual_price]])

# --- Streamlit App Config ---
st.set_page_config(page_title="Price Finder", page_icon="🛍️", layout="wide")
st.title("🛍️ Price Finder")
st.markdown("Enter a product brand and model to find its current price and prediction.")

# --- API Key ---
api_key = "d8a65400d0c0107a0276447f67f9149a15a97462caf453b95e702e07a63e3006"

# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    brand = st.text_input("Enter the Brand Name", placeholder="e.g., Apple")
with col2:
    model = st.text_input("Enter the Model", placeholder="e.g., iPhone 14")

# Store search results globally
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# --- Search Button ---
if st.button("🔍 Search for Prices", type="primary"):
    if not api_key:
        st.warning("Please enter your SerpApi API Key to proceed.")
    elif not brand or not model:
        st.warning("Please enter both a brand and a model.")
    else:
        search_query = f"{brand} {model}"
        st.info(f"Searching on Amazon.in for: **{search_query}**")
        try:
            params = {
                        "engine": "amazon",
                            "amazon_domain": "amazon.in",  # Force Amazon India site
                            "k": search_query,
                            "api_key": api_key
                        }

            search = requests.get("https://serpapi.com/search", params=params)
            search.raise_for_status()
            results = search.json()

            product_data = []
            if "organic_results" in results:
                for result in results.get("organic_results", []):
                    title = result.get("title", "")
                    link = result.get("link")
                    asin = result.get("asin")
                    thumbnail = result.get("thumbnail")
                    price_info = result.get("price")
                    if isinstance(price_info, dict):
                        current_price = price_info.get("raw", "Not available")
                    elif isinstance(price_info, str):
                        current_price = price_info
                    else:
                        current_price = "Not available"

                    rating = result.get("rating", "N/A")
                    reviews_count = result.get("reviews", "N/A")
                    delivery = result.get("delivery", "Not specified")

                    if title and link:
                        product_data.append({
                            "Image": thumbnail,
                            "Product Title": title,
                            "Price (INR)": current_price,
                            "Rating": rating,
                            "Reviews": reviews_count,
                            "Delivery": delivery,
                            "ASIN": asin,
                            "Link": link,
                            "Main Category": brand,
                            "Sub Category": model
                        })

            if product_data:
                st.session_state.search_results = pd.DataFrame(product_data)
                st.success("Products fetched! Click 'Predict Price' to see results.")
            else:
                st.warning("No products found for your search query on Amazon.in.")

        except Exception as e:
            st.error(f"Error: {e}")

# --- Predict Button ---
if st.session_state.search_results is not None and st.button("📊 Predict Price"):
    df = st.session_state.search_results.copy()
    predictions_xgb = []
    for _, row in df.iterrows():
        features = prepare_features(row)
        predictions_xgb.append(xgb_model.predict(features)[0])
    df["Predicted Price (XGB)"] = predictions_xgb

    def recommend_status(current_price, predicted_price):
        try:
            current_price_num = float(str(current_price).replace("₹", "").replace(",", "").strip())
        except:
            return "Unknown"
        if current_price_num < predicted_price * 0.9:
            return "Best Time to Buy"
        elif current_price_num > predicted_price * 1.1:
            return "Wait for Price Drop"
        else:
            return "Normal"

    df["Recommendation"] = [
        recommend_status(cp, pp) for cp, pp in zip(df["Price (INR)"], df["Predicted Price (XGB)"])
    ]
     # --- Apply color formatting ---
    def color_recommendation(val):
        if val == "Best Time to Buy":
            return "background-color: green; color: white;"
        elif val == "Wait for Price Drop":
            return "background-color: yellow; color: black;"
        elif val == "Unknown":
            return "background-color: red; color: white;"
        else:
            return ""

    st.subheader("✅ Predicted Prices")


    st.dataframe(
        styled_df = df.style.applymap(color_recommendation, subset=["Recommendation"])
        styled_df = styled_df.hide(axis="index")
    )
    # st.dataframe(df, hide_index=True, use_container_width=True)

# --- Show Company Stock Graph ---
if brand:
    st.subheader(f"📈 {brand} Stock Price Trend (via Yahoo Finance)")
    try:
        ticker = "AMZN" if brand.lower() == "amazon" else brand.upper()
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist.index, hist["Close"], label="Closing Price")
        ax.set_title(f"{ticker} Stock Price - Last 6 Months")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not fetch stock data for {brand}: {e}")

# import streamlit as st
# import requests
# import pandas as pd
# import os
# import json
# import pickle
# import numpy as np
# from sklearn.preprocessing import LabelEncoder  # Add this at the top


# import joblib

# # --- Load ML Model & Label Encoders ---
# xgb_model = joblib.load("price_predictor.pkl")  # Trained XGBoost model
# label_encoders = joblib.load("label_encoders.pkl")  # Dictionary of LabelEncoders

# # --- Function to preprocess data for model prediction ---
# def prepare_features(row):
#     try:
#         name_enc = label_encoders["name"].transform([row["Product Title"]])[0]
#     except:
#         name_enc = 0  # Fallback for unseen products
    
#     try:
#         main_cat = label_encoders["main_category"].transform([row.get("Main Category", "Unknown")])[0]
#     except:
#         main_cat = 0
    
#     try:
#         sub_cat = label_encoders["sub_category"].transform([row.get("Sub Category", "Unknown")])[0]
#     except:
#         sub_cat = 0

#     rating = float(row["Rating"]) if row["Rating"] != "N/A" else 0
#     reviews = float(row["Reviews"]) if row["Reviews"] != "N/A" else 0

#     try:
#         actual_price = float(str(row["Price (INR)"]).replace("₹", "").replace(",", "").strip())
#     except:
#         actual_price = 0

#     return np.array([[name_enc, main_cat, sub_cat, rating, reviews, actual_price]])

# # --- Main Streamlit App ---
# st.set_page_config(page_title="Amazon Price Finder", page_icon="🛍️", layout="wide")

# st.title("🛍️ Amazon.in Price Finder")
# st.markdown("Enter a product brand and model to find its current price on Amazon India.")

# # --- API Key ---
# api_key = "d8a65400d0c0107a0276447f67f9149a15a97462caf453b95e702e07a63e3006"

# # --- User Inputs ---
# col1, col2 = st.columns(2)
# with col1:
#     brand = st.text_input("Enter the Brand Name", placeholder="e.g., Apple")
# with col2:
#     model = st.text_input("Enter the Model", placeholder="e.g., iPhone 14")

# # --- Search and Display Logic ---
# if st.button("🔍 Search for Prices", type="primary"):
#     if not api_key:
#         st.warning("Please enter your SerpApi API Key to proceed.")
#     elif not brand or not model:
#         st.warning("Please enter both a brand and a model.")
#     else:
#         search_query = f"{brand} {model}"
#         st.info(f"Searching on amazon.in for: **{search_query}**")

#         with st.spinner("Fetching data from Amazon India... Please wait. ⏳"):
#             try:
#                 # API Request Parameters
#                 params = {
#                     "engine": "amazon",
#                     "k": search_query,
#                     "country": "in",
#                     "api_key": api_key
#                 }

#                 search = requests.get("https://serpapi.com/search", params=params)
#                 search.raise_for_status()
#                 results = search.json()

#                 product_data = []
#                 if "organic_results" in results:
#                     filtered_results = []

#                     for result in results.get("organic_results", []):
#                         title = result.get("title", "")
#                         link = result.get("link")
#                         asin = result.get("asin")
#                         thumbnail = result.get("thumbnail")

#                         # Price extraction
#                         price_info = result.get("price")
#                         if isinstance(price_info, dict):
#                             current_price = price_info.get("raw", "Not available")
#                         elif isinstance(price_info, str):
#                             current_price = price_info
#                         else:
#                             current_price = "Not available"

#                         rating = result.get("rating", "N/A")
#                         reviews_count = result.get("reviews", "N/A")
#                         delivery = result.get("delivery", "Not specified")

#                         if title and link:
#                             filtered_results.append({
#                                 "Image": thumbnail,
#                                 "Product Title": title,
#                                 "Price (INR)": current_price,
#                                 "Rating": rating,
#                                 "Reviews": reviews_count,
#                                 "Delivery": delivery,
#                                 "ASIN": asin,
#                                 "Link": link,
#                                 "Main Category": brand,  # Using brand as placeholder category
#                                 "Sub Category": model    # Using model as placeholder sub-category
#                             })

#                     # Prefer original products
#                     original_only = [p for p in filtered_results if "renewed" not in p["Product Title"].lower() and "case" not in p["Product Title"].lower()]

#                     if original_only:
#                         product_data.append(original_only[0])
#                     elif filtered_results:
#                         product_data.append(filtered_results[0])

#                 if product_data:
#                     df = pd.DataFrame(product_data)

#                     # Predictions
#                     predictions_xgb = []
#                     for _, row in df.iterrows():
#                         features = prepare_features(row)
#                         predictions_xgb.append(xgb_model.predict(features)[0])

#                     df["Predicted Price (XGB)"] = predictions_xgb

#                     # Recommendation logic
#                     def recommend_status(current_price, predicted_price):
#                         try:
#                             current_price_num = float(str(current_price).replace("₹", "").replace(",", "").strip())
#                         except:
#                             return "Unknown"
#                         if current_price_num < predicted_price * 0.9:
#                             return "Best Time to Buy"
#                         elif current_price_num > predicted_price * 1.1:
#                             return "Wait for Price Drop"
#                         else:
#                             return "Normal"

#                     df["Recommendation"] = [
#                         recommend_status(cp, pp) for cp, pp in zip(df["Price (INR)"], df["Predicted Price (XGB)"])
#                     ]

#                     st.subheader("✅ Search Results from Amazon.in")
#                     st.dataframe(
#                         df,
#                         column_config={
#                             "Image": st.column_config.ImageColumn("Preview"),
#                             "Product Title": st.column_config.TextColumn("Product", width="large"),
#                             "Price (INR)": st.column_config.TextColumn("Price"),
#                             "Rating": st.column_config.NumberColumn("⭐", format="%.1f"),
#                             "Reviews": st.column_config.NumberColumn("Reviews Count"),
#                             "Link": st.column_config.LinkColumn("Buy on Amazon", display_text="Go to Page 🔗"),
#                         },
#                         hide_index=True,
#                         use_container_width=True
#                     )
#                 else:
#                     st.warning("No products found for your search query on Amazon.in.")

#             except requests.exceptions.HTTPError as err:
#                 st.error(f"API Error: {err}. Please check your API key or search terms.")
#             except Exception as e:
#                 st.error(f"An unexpected error occurred: {e}")

# import streamlit as st
# import requests
# import pandas as pd
# import os
# import json
# import pickle
# import numpy as np
# from sklearn.preprocessing import LabelEncoder  # Add this at the top

# # --- Load ML Models ---
# xgb_model = pickle.load(open("price_predictor.pkl", "rb"))
# rf_model = pickle.load(open("label_encoders.pkl", "rb"))

# # --- Function to preprocess data for model prediction ---
# def prepare_features(row):
#     return np.array([[
#         len(row["Product Title"]),  # simple proxy for title length
#         float(row["Rating"]) if row["Rating"] != "N/A" else 0,
#         float(row["Reviews"]) if row["Reviews"] != "N/A" else 0
#     ]])  # You can add more features later from your training set

# # --- Main Streamlit App ---
# st.set_page_config(page_title="Amazon Price Finder", page_icon="🛍️", layout="wide")

# st.title("🛍️ Amazon.in Price Finder")
# st.markdown("Enter a product brand and model to find its current price on Amazon India.")

# # --- API Key (Replace with your own SerpApi key) ---
# api_key = "d8a65400d0c0107a0276447f67f9149a15a97462caf453b95e702e07a63e3006"

# # --- User Inputs ---
# col1, col2 = st.columns(2)
# with col1:
#     brand = st.text_input("Enter the Brand Name", placeholder="e.g., Apple")
# with col2:
#     model = st.text_input("Enter the Model", placeholder="e.g., iPhone 14")

# # --- Search and Display Logic ---
# if st.button("🔍 Search for Prices", type="primary"):
#     if not api_key:
#         st.warning("Please enter your SerpApi API Key to proceed.")
#     elif not brand or not model:
#         st.warning("Please enter both a brand and a model.")
#     else:
#         search_query = f"{brand} {model}"
#         st.info(f"Searching on amazon.in for: **{search_query}**")

#         with st.spinner("Fetching data from Amazon India... Please wait. ⏳"):
#             try:
#                 # --- API Request Parameters ---
#                 params = {
#                     "engine": "amazon",
#                     "k": search_query,
#                     "country": "in",
#                     "api_key": api_key
#                 }

#                 search = requests.get("https://serpapi.com/search", params=params)
#                 search.raise_for_status()
#                 results = search.json()

#                 product_data = []
#                 if "organic_results" in results:
#                     filtered_results = []

#                     for result in results.get("organic_results", []):
#                         title = result.get("title", "")
#                         link = result.get("link")
#                         asin = result.get("asin")
#                         thumbnail = result.get("thumbnail")

#                         # --- Safe price extraction ---
#                         price_info = result.get("price")
#                         if isinstance(price_info, dict):
#                             current_price = price_info.get("raw", "Not available")
#                         elif isinstance(price_info, str):
#                             current_price = price_info
#                         else:
#                             current_price = "Not available"

#                         rating = result.get("rating", "N/A")
#                         reviews_count = result.get("reviews", "N/A")
#                         delivery = result.get("delivery", "Not specified")

#                         if title and link:
#                             filtered_results.append({
#                                 "Image": thumbnail,
#                                 "Product Title": title,
#                                 "Price (INR)": current_price,
#                                 "Rating": rating,
#                                 "Reviews": reviews_count,
#                                 "Delivery": delivery,
#                                 "ASIN": asin,
#                                 "Link": link
#                             })

#                     # Try to find original (non-renewed) product first
#                     original_only = [p for p in filtered_results if "renewed" not in p["Product Title"].lower() and "case" not in p["Product Title"].lower()]

#                     if original_only:
#                         product_data.append(original_only[0])  # Take first original match
#                     elif filtered_results:
#                         product_data.append(filtered_results[0])  # Fallback: first available



#                 if product_data:

#                     df = pd.DataFrame(product_data)
#                     # --- Predict prices using models ---
#                     predictions_xgb = []
#                     predictions_rf = []

#                     for _, row in df.iterrows():
#                         features = prepare_features(row)
#                         predictions_xgb.append(xgb_model.predict(features)[0])
#                         predictions_rf.append(rf_model.predict(features)[0])

#                     df["Predicted Price (XGB)"] = predictions_xgb
#                     df["Predicted Price (RF)"] = predictions_rf

#                     # --- Price Recommendation ---
#                     def recommend_status(current_price, predicted_price):
#                         try:
#                             current_price_num = float(str(current_price).replace("₹", "").replace(",", "").strip())
#                         except:
#                             return "Unknown"
#                         if current_price_num < predicted_price * 0.9:
#                             return "Best Time to Buy"
#                         elif current_price_num > predicted_price * 1.1:
#                             return "Wait for Price Drop"
#                         else:
#                             return "Normal"

#                     df["Recommendation"] = [
#                         recommend_status(cp, pp) for cp, pp in zip(df["Price (INR)"], df["Predicted Price (XGB)"])
#                     ]

#                     st.subheader("✅ Search Results from Amazon.in")

#                     st.dataframe(
#                         df,
#                         column_config={
#                             "Image": st.column_config.ImageColumn("Preview"),
#                             "Product Title": st.column_config.TextColumn("Product", width="large"),
#                             "Price (INR)": st.column_config.TextColumn("Price"),
#                             "Rating": st.column_config.NumberColumn("⭐", format="%.1f"),
#                             "Reviews": st.column_config.NumberColumn("Reviews Count"),
#                             "Link": st.column_config.LinkColumn("Buy on Amazon", display_text="Go to Page 🔗"),
#                         },
#                         hide_index=True,
#                         use_container_width=True
#                     )
#                 else:
#                     st.warning("No products found for your search query on Amazon.in.")

#             except requests.exceptions.HTTPError as err:
#                 st.error(f"API Error: {err}. Please check your API key or search terms.")
#             except Exception as e:
#                 st.error(f"An unexpected error occurred: {e}")

# import streamlit as st
# import requests
# import pandas as pd

# # --- Streamlit App Config ---
# st.set_page_config(page_title="Amazon Price Finder", page_icon="🛍️", layout="wide")

# st.title("🛍️ Amazon Price Finder")
# st.markdown("Enter a product brand and model to find its current price on Amazon.")

# # --- User Inputs ---
# api_key = "d8a65400d0c0107a0276447f67f9149a15a97462caf453b95e702e07a63e3006"  # For production, use st.secrets

# col1, col2 = st.columns(2)
# with col1:
#     brand = st.text_input("Enter the Brand Name", placeholder="e.g., Sony")
# with col2:
#     model = st.text_input("Enter the Model", placeholder="e.g., WH-1000XM5")

# # --- Search Action ---
# if st.button("🔍 Search for Prices", type="primary"):
#     if not api_key:
#         st.warning("Please enter your SerpApi API Key to proceed.")
#     elif not brand or not model:
#         st.warning("Please enter both a brand and a model.")
#     else:
#         search_query = f"{brand} {model}"
#         st.info(f"Searching for: **{search_query}**")

#         with st.spinner("Fetching data from Amazon... Please wait. ⏳"):
#             try:
#                 # --- API Call ---
#                 params = {
#                     "engine": "amazon",
#                     "k": search_query,  # Correct parameter for keyword
#                     "api_key": api_key
#                 }

#                 response = requests.get("https://serpapi.com/search", params=params)
#                 response.raise_for_status()
#                 results = response.json()

#                 # --- Parse Results ---
#                 product_data = []

#                 for result in results.get("organic_results", []):
#                     title = result.get("title")
#                     link = result.get("link")
#                     thumbnail = result.get("thumbnail")

#                     price_info = result.get("price")
#                     if isinstance(price_info, dict):
#                         current_price = price_info.get("raw", "Not available")
#                     elif isinstance(price_info, str):
#                         current_price = price_info
#                     else:
#                         current_price = "Not available"

#                     if title and link:
#                         product_data.append({
#                             "Image": thumbnail,
#                             "Product Title": title,
#                             "Current Price": current_price,
#                             "Link": link
#                         })

#                 # --- Display Results ---
#                 if product_data:
#                     df = pd.DataFrame(product_data)

#                     st.subheader("✅ Search Results")
#                     st.dataframe(
#                         df,
#                         column_config={
#                             "Image": st.column_config.ImageColumn("Preview", width="small"),
#                             "Product Title": st.column_config.TextColumn("Product", width="large"),
#                             "Current Price": st.column_config.TextColumn("Price"),
#                             "Link": st.column_config.LinkColumn(
#                                 "Buy on Amazon", display_text="Go to Page 🔗", width="medium"
#                             ),
#                         },
#                         hide_index=True,
#                         use_container_width=True
#                     )
#                 else:
#                     st.warning("No products found for your search query. Please try different terms.")

#             except requests.exceptions.HTTPError as http_err:
#                 st.error(f"API Error: {http_err}. Please check your API key or search terms.")
#             except Exception as e:
#                 st.error(f"An unexpected error occurred: {e}")

# import streamlit as st
# import requests
# import pandas as pd
# import os

# # --- Main Streamlit App ---

# st.set_page_config(page_title="Amazon Price Finder", page_icon="🛍️", layout="wide")

# st.title("🛍️ Amazon Price Finder")
# st.markdown("Enter a product brand and model to find its current price on Amazon.")

# # --- User Inputs ---

# # For better security, it's recommended to use st.secrets for the API key
# # https://docs.streamlit.io/library/advanced-features/secrets-management
# # For this example, we'll use a text input.
# api_key = "d8a65400d0c0107a0276447f67f9149a15a97462caf453b95e702e07a63e3006"
# col1, col2 = st.columns(2)
# with col1:
#     brand = st.text_input("Enter the Brand Name", placeholder="e.g., Sony")
# with col2:
#     model = st.text_input("Enter the Model", placeholder="e.g., WH-1000XM5")

# # --- Search and Display Logic ---

# if st.button("🔍 Search for Prices", type="primary"):
#     # Input validation
#     if not api_key:
#         st.warning("Please enter your SerpApi API Key to proceed.")
#     elif not brand or not model:
#         st.warning("Please enter both a brand and a model.")
#     else:
#         search_query = f"{brand} {model}"
#         st.info(f"Searching for: **{search_query}**")

#         with st.spinner("Fetching data from Amazon... Please wait. ⏳"):
#             try:
#                 # --- API Request Parameters ---
#                 params = {
#                     "engine": "amazon",
#                     "k": search_query,
#                     "api_key": api_key
#                 }

#                 # --- Make the API Call ---
#                 search = requests.get("https://serpapi.com/search", params=params)
#                 search.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
#                 results = search.json()

#                 product_data = []

#                 # --- Process the Response ---
#                 if "organic_results" in results:
#                     for result in results.get("organic_results", []):
#                         # Safely get data using .get() to avoid errors if a key is missing
#                         title = result.get("title")
#                         price_info = result.get("price")
#                         link = result.get("link")
#                         thumbnail = result.get("thumbnail")

#                         # The price is often in a nested dictionary
#                         current_price = price_info.get("raw") if price_info else "Not available"

#                         # Ensure essential data is present before adding to the list
#                         if title and link:
#                             product_data.append({
#                                 "Image": thumbnail,
#                                 "Product Title": title,
#                                 "Current Price": current_price,
#                                 "Link": link
#                             })

#                 # --- Display Results ---
#                 if product_data:
#                     df = pd.DataFrame(product_data)
#                     st.subheader("✅ Search Results")

#                     # Display the data in a Streamlit dataframe
#                     st.dataframe(
#                         df,
#                         column_config={
#                             "Image": st.column_config.ImageColumn("Preview", width="small"),
#                             "Product Title": st.column_config.TextColumn("Product", width="large"),
#                             "Current Price": st.column_config.TextColumn("Price"),
#                             "Link": st.column_config.LinkColumn(
#                                 "Buy on Amazon",
#                                 display_text="Go to Page 🔗",
#                                 width="medium"
#                             ),
#                         },
#                         hide_index=True,
#                         use_container_width=True
#                     )
#                 else:
#                     st.warning("No products found for your search query. Please try different terms.")

#             except requests.exceptions.HTTPError as err:
#                 st.error(f"API Error: {err}. Please check your API key or search terms.")
#             except Exception as e:
#                 st.error(f"An unexpected error occurred: {e}")


# import streamlit as st
# import pandas as pd
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# import time

# # --- NEW Scraper Function for Standard Google Search ---
# def scrape_google_search_results(query: str, max_results: int = 5):
#     """
#     Scrapes the main Google search page, targeting the product shopping boxes.
#     """
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#     options.add_argument('--log-level=3')
#     options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    
#     try:
#         driver = webdriver.Chrome(options=options)
#         # URL for a standard Google search
#         safe_query = query.replace(' ', '+')
#         url = f"https://www.google.com/search?q={safe_query}"
#         driver.get(url)
#     except Exception as e:
#         st.error(f"Error starting browser: {e}")
#         return []

#     scraped_data = []
    
#     # --- THIS IS THE UPDATED PART ---
#     # We now look for the class name of sponsored product units: 'pla-unit'
#     # Note: These class names can change frequently.
#     product_containers = driver.find_elements(By.CSS_SELECTOR, "div.pla-unit")

#     for container in product_containers[:max_results]:
#         try:
#             # Extract data using the new class names for this type of result
#             model_name = container.find_element(By.CSS_SELECTOR, "div.pla-unit-title-link").text
#             price_str = container.find_element(By.CSS_SELECTOR, "div.e10twf").text
#             price_inr = int(''.join(filter(str.isdigit, price_str)))
#             seller = container.find_element(By.CSS_SELECTOR, "div.vJ931").text
#             image_url = container.find_element(By.CSS_SELECTOR, "div.Gor6zc img").get_attribute('src')
            
#             scraped_data.append({
#                 "Scraped Listing": model_name,
#                 "Price (INR)": price_inr,
#                 "Seller": seller,
#                 "Image URL": image_url
#             })
#         except Exception:
#             continue
            
#     driver.quit()
#     return scraped_data

# # --- Streamlit User Interface (with updated text) ---

# st.set_page_config(layout="wide", page_title="Phone Price Finder")
# st.title("📱 Phone Price & Image Finder")
# st.markdown("Enter phone features in the sidebar, and we'll scrape the **main Google search page** for live prices and images.")

# st.sidebar.header("Enter Phone Features")

# brand_options = ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Nothing", "Motorola"]
# ram_options = ["Not Specified", "4GB", "6GB", "8GB", "12GB", "16GB"]
# storage_options = ["Not Specified", "64GB", "128GB", "256GB", "512GB", "1TB"]

# input_brand = st.sidebar.selectbox("Select Brand", options=brand_options)
# input_model = st.sidebar.text_input("Enter Model Name", "iPhone 15")
# input_ram = st.sidebar.selectbox("Select RAM", options=ram_options)
# input_storage = st.sidebar.selectbox("Select Storage (ROM)", options=storage_options)
# input_ss = st.sidebar.text_input("Enter Screen Size (Optional)", "6.1 inches")
# input_battery = st.sidebar.text_input("Enter Battery (Optional)", "3274 mAh")

# if st.sidebar.button("Find Price & Image"):
#     if not input_model:
#         st.warning("Please enter a model name to search.")
#     else:
#         search_query = f"{input_brand} {input_model} price in India"
        
#         with st.spinner(f"🔎 Performing a Google search for '{search_query}'..."):
#             # Call the new scraper function
#             scraped_results = scrape_google_search_results(search_query)

#         if scraped_results:
#             st.success(f"Found {len(scraped_results)} product listings for '{search_query}'!")
            
#             first_result = scraped_results[0]
#             st.markdown("---")
#             st.subheader("Top Result")
            
#             col1, col2 = st.columns([1, 2])
#             with col1:
#                 st.image(first_result['Image URL'], caption=first_result['Scraped Listing'])
#             with col2:
#                 st.metric(label="Price", value=f"₹ {first_result['Price (INR)']:,}")
#                 st.write(f"**Full Title:** {first_result['Scraped Listing']}")
#                 st.write(f"**Seller:** {first_result['Seller']}")
            
#             st.markdown("---")
            
#             display_list = []
#             for result in scraped_results:
#                 display_list.append({
#                     'Input Brand': input_brand,
#                     'Input Model': input_model,
#                     'RAM': input_ram,
#                     'Storage': input_storage,
#                     'Price (INR)': result['Price (INR)'],
#                     'Seller': result['Seller'],
#                     'Scraped Listing Title': result['Scraped Listing']
#                 })
            
#             df = pd.DataFrame(display_list)
#             st.subheader("Full Results Table")
#             st.dataframe(df)

#         else:
#             st.error(f"Could not find any product shopping boxes for '{search_query}'. Please try a different model name.")
