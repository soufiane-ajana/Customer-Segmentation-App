# 🌌 Customer Insights Pro: Predictive Segmentation Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

**Live Demo:** [👉 Click here to view the live application : https://customer-segmentation-app-9hzmzuml8kxq2gavgwoawk.streamlit.app/#profil-identifie-classe-moyenne-coeur-de-cible]

## 📌 Project Overview
This project is an end-to-end **Data Science & Machine Learning application** designed to help marketing and management teams transition from intuition-based decisions to a data-driven strategy. 

It features a real-time predictive dashboard that classifies incoming customers into distinct behavioral segments based on their Annual Income and Spending Score, allowing for immediate, automated, and personalized marketing actions.

## 🧠 Machine Learning Architecture
The core engine relies on a hybrid Machine Learning approach:
1. **Unsupervised Learning (K-Means):** Used historically on the customer database to identify the natural mathematical structure of the data and isolate 5 distinct clusters (validated via the Elbow Method).
2. **Supervised Learning (Decision Tree Classifier):** Trained on the K-Means results to create an interpretable, lightning-fast prediction algorithm (`.pkl` model) capable of classifying any new customer in real-time with >95% accuracy.

## ✨ Key Features
* **Interactive UI:** Built with Streamlit, featuring a modern Dark Mode interface and smooth CSS animations.
* **Real-Time Prediction:** Adjust the sliders to simulate a new prospect and instantly see their predicted segment.
* **Dynamic Data Visualization:** Powered by `Plotly Express`, the scatter plot updates in real-time, placing the simulated prospect directly on the spatial map.
* **Actionable Business Insights:** Each prediction comes with a tailored marketing strategy (e.g., VIP Concierge, Push Marketing, Retention campaigns) to optimize CAC and LTV.

## 🛠️ Technology Stack
* **Language:** Python 3.11
* **Frontend/Deployment:** Streamlit, Streamlit Community Cloud
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (KMeans, DecisionTreeClassifier)
* **Data Visualization:** Plotly, Matplotlib, Seaborn

## 🚀 How to Run Locally

If you wish to run this project on your own machine:

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Customer-Segmentation-App.git](https://github.com/soufiane-ajana/Customer-Segmentation-App.git)
2. Navigate to the project directory:
   cd Customer-Segmentation-App
3. Install the required dependencies:
   pip install -r requirements.txt
4. Run the Streamlit server:
   streamlit run app.py
