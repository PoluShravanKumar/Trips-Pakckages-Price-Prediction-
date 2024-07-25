import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pickle


st.set_page_config(page_title="Trips package Price Classification", page_icon=":package:", layout="centered")


st.image(r"Screenshot 2024-07-25 185502.png")
st.image(r"illustration-of-travel-using-as-business-web-template-agency-free-vector.jpg")


st.title("Trips  Price Classification Project")
st.subheader("By Shravan Kumar Polu")
st.markdown("Predict the price range of a Trips based on its specifications.")


model = pickle.load(open(r"Trip_Price.pkl", "rb"))

cities = [
    'Goa', 'Port Blair', 'Mussoorie', 'Coorg', 'Manali', 'Munnar',
    'Calangute', 'Rishikesh', 'Gangtok', 'Leh', 'Kaziranga', 'Shimla',
    'Katra', 'Udaipur', 'Ooty', 'Srinagar', 'Cochin', 'Guwahati',
    'Jaipur', 'Darjeeling', 'Lansdowne', 'Kochi', 'Pahalgam',
    'Mount Abu', 'Jim Corbett', 'Wayanad', 'Pelling', 'Mahabaleshwar',
    'Kodaikanal', 'Ranthambore', 'Lonavala', 'Jaisalmer', 'Shillong',
    'Alleppey', 'Kasol', 'Dalhousie', 'Athirapally', 'Haridwar',
    'Jodhpur', 'Kumarakom', 'New Delhi', 'Auli', 'Chikmagalur',
    'Tirupati', 'Mcleodganj', 'Ootacamund', 'Candolim', 'Vaikom',
    'Pachmarhi', 'Dehradun', 'Calicut', 'Mumbai', 'Madurai', 'Shirdi',
    'Jamnagar', 'Dharamshala', 'Dwarka', 'Mysore', 'Ahmedabad', 'Bhuj',
    'Bhalukpong', 'Puri', 'Nainital', 'Kovalam', 'Sasan Gir', 'Agra',
    'Cherrapunjee', 'sariska', 'Hyderabad', 'Kasauli', 'Kanyakumari',
    'Amritsar', 'Kaziranga National Park', 'Bhubaneshwar', 'Matheran',
    'Udupi', 'Rameshwaram', 'Chennai', 'Dimapur', 'POKHARA',
    'Mahabalipuram', 'Ranikhet', 'Chopta', 'Panchgani', 'Pondicherry',
    'Dhanaulti', 'Kausani', 'Kabini', 'Bhimtal', 'Kalimpong', 'Kutch',
    'Rameswaram', 'Indore', 'Athirappilly', 'lavasa', 'Parwanoo',
    'Hampi', 'Jabalpur', 'Maheshwar', 'Aurangabad'
]

city_to_code = {city: index for index, city in enumerate(cities)}
code_to_city = {index: city for city, index in city_to_code.items()}

duration_days = st.slider("Duration of Days", min_value=1, max_value=13, step=1)
destination_cities_count = st.slider("Number of Destination Cities", min_value=1, max_value=8, step=1)
hotel_ratings = st.slider("Hotel Ratings", min_value=1, max_value=5, step=1)
discount_percentage = st.slider("Discount Percentage", min_value=0, max_value=39, step=1)
city_name = st.selectbox("Select City", options=list(city_to_code.keys()))
city_code = city_to_code[city_name]


st.write("### Input Values")
st.write(f"Duration of Days: {duration_days}")
st.write(f"Number of Destination Cities: {destination_cities_count}")
st.write(f"Hotel Ratings: {hotel_ratings}")
st.write(f"Discount Percentage: {discount_percentage}")
st.write(f"City Code: {city_code}")

if st.button("Predict"):
    features = np.array([[duration_days, destination_cities_count, hotel_ratings, discount_percentage, city_code]])
   
    price_category = model.predict(features)[0]

    
    price_categories = {
        0: "Low Cost: Below ₹10,000",
        1: "Affordable Mid-range: ₹10,000 - ₹20,000",
        2: "Upper Mid-range: ₹20,000 - ₹35,000",
        3: "Premium: ₹35,000 - ₹50,000",
        4: "Luxury: Above ₹50,000"
    }

   
    price_category_desc = price_categories.get(price_category, "Unknown Category")
    
 
    st.write("### Prediction")
    st.write(f"Price Category: {price_category_desc}")
