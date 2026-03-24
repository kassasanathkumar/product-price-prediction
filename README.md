# product-price-prediction


# 🛍️ Product Price Prediction System 

## 🔍 Overview

This project is a **Machine Learning-based Product Price Prediction System** that predicts the expected price of products using real-time search results.

Unlike traditional systems, this project **does NOT use web scraping**. Instead, it uses:

* **SerpAPI** to fetch product data from Amazon
* A trained **XGBoost model** to predict prices

It helps users determine whether it is the **right time to buy a product**.

---

## 🎯 Objectives

* Predict product prices using ML
* Avoid web scraping by using APIs
* Provide smart recommendations:

  * ✅ Best Time to Buy
  * ⏳ Wait for Price Drop
  * ⚖️ Normal Price

---

## 🧠 Technologies Used

* **Frontend/UI:** Streamlit
* **Backend:** Python
* **Machine Learning Model:** XGBoost
* **Libraries:**

  * pandas, numpy
  * scikit-learn
  * joblib
  * matplotlib
* **API Used:** SerpAPI (Amazon search results)

---

## ⚙️ How It Works

### 1. User Input

* Enter **Brand** and **Model**

### 2. Data Fetching (No Scraping)

* Uses **SerpAPI** to fetch product details from Amazon

### 3. Feature Engineering

From the fetched data:

* Product title
* Category & sub-category
* Rating
* Reviews
* Current price

### 4. Prediction

* Data is passed into trained **XGBoost model**
* Model predicts expected product price

### 5. Recommendation System

Based on comparison:

* If current price < predicted price → ✅ Best Time to Buy
* If current price > predicted price → ⏳ Wait
* Else → ⚖️ Normal

---

## 📊 Features

* 🔍 Real-time product search (API-based)
* 📈 Price prediction using ML
* 💡 Smart buying recommendations
* 📊 Interactive UI using Streamlit
* 📉 Stock trend visualization (Yahoo Finance)

---

## 📁 Project Structure

```bash
product-price-prediction/
│
├── app.py                      # Main Streamlit app
├── price_predictor.pkl         # Trained XGBoost model
├── label_encoders.pkl          # Encoders for categorical data
├── requirements.txt            # Dependencies
└── README.md
```

---

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/product-price-prediction.git
cd product-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API Key

Replace your SerpAPI key in `app.py`:

```python
api_key = "YOUR_SERPAPI_KEY"
```

### 4. Run Application

```bash
streamlit run app.py
```

---

## 📌 Sample Workflow

1. Enter:

   * Brand: Apple
   * Model: iPhone 14

2. System:

   * Fetches Amazon results via API
   * Predicts price using ML model
   * Displays recommendation

---

## 🧪 Model Details

* Model: **XGBoost Regressor**
* Input Features:

  * Encoded product title
  * Category
  * Rating
  * Reviews
  * Current price
* Output:

  * Predicted price

---

## 📈 Additional Feature

* Displays **company stock trends** using Yahoo Finance
* Helps correlate product pricing with company performance

---

## ⚠️ Important Notes

* ❌ No web scraping is used
* ✅ Fully API-based system
* Requires **internet connection**
* SerpAPI has usage limits (free tier available)

---

## 🌍 Applications

* E-commerce analytics
* Smart shopping assistants
* Price comparison tools
* Consumer decision support systems

---

## 📌 Conclusion

This project provides a modern approach to price prediction by combining:

* Real-time API data
* Machine learning
* User-friendly interface

It avoids the complexity of web scraping while still delivering accurate and useful predictions.

---

## 👨‍💻 Author

**Sanath Kumar**
B.Tech AIDS(CSE)

