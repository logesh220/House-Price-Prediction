
import joblib
import pandas as pd
import streamlit as st


# STEP 1: Configure the website
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# STEP 2: Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("models/house_price_model.pkl")


model = load_model()


# STEP 3: Website heading
st.title("🏠 House Price Prediction")

st.write(
    "Estimate a property's price using machine learning "
    "based on its location, size, and features."
)

st.divider()


# STEP 4: Collect property information
st.header("Enter Property Details")

col1, col2 = st.columns(2)


# LOCATION AND PROPERTY DETAILS
with col1:
    st.subheader("📍 Location and Property")

    city = st.selectbox(
        "City",
        [
            "Ahmedabad",
            "Bangalore",
            "Chandigarh",
            "Chennai",
            "Delhi",
            "Hyderabad",
            "Jaipur",
            "Kolkata",
            "Mumbai",
            "Pune"
        ]
    )

    locality_type = st.selectbox(
        "Locality Type",
        ["Affordable", "Mid-Range", "Premium"]
    )

    property_type = st.selectbox(
        "Property Type",
        [
            "Apartment",
            "Builder Floor",
            "Independent House",
            "Villa"
        ]
    )

    bhk = st.number_input(
        "Number of Bedrooms (BHK)",
        min_value=1,
        max_value=20,
        value=2
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=20,
        value=2
    )

    furnishing_status = st.selectbox(
        "Furnishing Status",
        ["Furnished", "Semi-Furnished", "Unfurnished"]
    )


# AREA AND AMENITIES
with col2:
    st.subheader("📐 Area and Amenities")

    super_area = st.number_input(
        "Super Area (sq. ft.)",
        min_value=100,
        max_value=50000,
        value=1000
    )

    carpet_area = st.number_input(
        "Carpet Area (sq. ft.)",
        min_value=100,
        max_value=50000,
        value=800
    )

    floor_number = st.number_input(
        "Floor Number",
        min_value=0,
        max_value=200,
        value=1
    )

    total_floors = st.number_input(
        "Total Floors in Building",
        min_value=1,
        max_value=200,
        value=5
    )

    age = st.number_input(
        "Property Age (years)",
        min_value=0,
        max_value=200,
        value=5
    )

    parking = st.number_input(
        "Number of Parking Spaces",
        min_value=0,
        max_value=20,
        value=1
    )


# ADDITIONAL FEATURES
st.divider()

st.subheader("Additional Features")

col3, col4, col5 = st.columns(3)

with col3:
    lift_available = st.selectbox(
        "Lift Available?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col4:
    gated_community = st.selectbox(
        "Gated Community?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col5:
    distance_metro = st.number_input(
        "Distance to Metro (km)",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.5
    )

distance_city_center = st.number_input(
    "Distance to City Center (km)",
    min_value=0.0,
    max_value=200.0,
    value=5.0,
    step=0.5
)


# STEP 5: Predict property price
st.divider()

if st.button("Predict House Price", type="primary"):

    if carpet_area > super_area:
        st.error(
            "Carpet area should not exceed super area. "
            "Please check the entered values."
        )

    elif floor_number > total_floors:
        st.error(
            "Floor number cannot exceed the total number of floors."
        )

    else:
        # Prepare input data using the same feature names
        # used when training the machine learning model.
        input_data = pd.DataFrame([{
            "City": city,
            "Locality_Type": locality_type,
            "Property_Type": property_type,
            "BHK": bhk,
            "Bathrooms": bathrooms,
            "Super_Area_SqFt": super_area,
            "Carpet_Area_SqFt": carpet_area,
            "Floor_Number": floor_number,
            "Total_Floors": total_floors,
            "Age_of_Property": age,
            "Furnishing_Status": furnishing_status,
            "Parking": parking,
            "Lift_Available": lift_available,
            "Gated_Community": gated_community,
            "Distance_to_Metro_km": distance_metro,
            "Distance_to_City_Center_km": distance_city_center
        }])

        # Generate prediction
        predicted_price = model.predict(input_data)[0]

        # Display prediction result
        st.success("Price prediction completed!")

        st.subheader("💰 Estimated Property Price")

        st.metric(
            label="Predicted Price",
            value=f"₹ {predicted_price:.2f} Lakhs",
            help=(
                "This is a machine learning estimate, "
                "not a verified market valuation."
            )
        )

        st.info(
            f"Approximately ₹{predicted_price * 100000:,.0f}"
        )

        # Display property summary
        st.subheader("Property Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        summary_col1.metric("Bedrooms", bhk)
        summary_col2.metric("Bathrooms", bathrooms)
        summary_col3.metric(
            "Super Area",
            f"{super_area:,} sq. ft."
        )

        st.write(f"**City:** {city}")
        st.write(f"**Property Type:** {property_type}")
        st.write(f"**Locality Type:** {locality_type}")
        st.write(f"**Furnishing:** {furnishing_status}")

        st.caption(
            "This estimate depends on the training dataset and "
            "model performance. Actual market prices may differ."
        )


# STEP 6: Footer
st.divider()

st.caption(
    "House Price Prediction | Machine Learning Project"
)

