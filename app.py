import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define a username and password
valid_username = "vaishali"
valid_password = "vaishali"

# Define a custom class for session state management
class SessionState:
    def __init__(self):
        self.logged_in = False

# Create an instance of the SessionState class
session_state = SessionState()

# Title of the app
st.title('Flight Price Prediction App')

# Function to check login credentials
def login():
    st.sidebar.header('Login')
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == valid_username and password == valid_password:
            st.session_state.logged_in = True
            st.success(f"Hi, {username}! Login successful!")
        else:
            st.warning("Invalid username or password. Please try again.")

# Check if the user is logged in
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# If the user is not logged in, display the login page
if not st.session_state.logged_in:
    login()

# If the user is logged in, display the main content
if st.session_state.logged_in:
    # Path to your flight price dataset (replace with your dataset)
    file_path = 'datasets/Clean_Dataset.csv'

    # Read the flight price dataset
    df = pd.read_csv(file_path)

    # Sidebar for filters and custom query
    st.sidebar.header('Filter and Search Options')

    # Selectbox for airline company
    selected_airline = st.sidebar.selectbox('Select Airline Company', options=['All']+list(df['airline'].unique()))

    # Selectbox for source city
    selected_source = st.sidebar.selectbox('Select Source City', options=['All']+list(df['source_city'].unique()))

    # Selectbox for destination city
    selected_destination = st.sidebar.selectbox('Select Destination City', options=['All']+list(df['destination_city'].unique()))

    # Custom Query Input
    st.sidebar.subheader('Custom Query')
    custom_query = st.sidebar.text_input("Enter your query (e.g., duration > 2.0)")

    # Filtering the dataframe based on selection
    if selected_airline != 'All':
        df = df[df['airline'] == selected_airline]

    if selected_source != 'All':
        df = df[df['source_city'] == selected_source]

    if selected_destination != 'All':
        df = df[df['destination_city'] == selected_destination]

    # Applying Custom Query
    if custom_query:
        try:
            df = df.query(custom_query)
        except Exception as e:
            st.error(f"Error in query: {e}")

    # Display the filtered dataframe
    st.write("Filtered Data", df)

    # Features for prediction based on your dataset
    features = ['duration', 'days_left']  # Adjust based on your dataset

    # Target variable
    target = 'price'

    # Split data into train and test sets
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate RMSE as an example of evaluation
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Display predicted prices and evaluation metric
    st.header("Price Prediction")
    st.write("Predicted Prices:", y_pred)

    # Input form for user to enter flight features for price prediction
    st.header("Enter Flight Details for Price Prediction")
    duration = st.slider("Duration (hours)", min_value=0.0, max_value=10.0, step=0.01)
    days_left = st.slider("Days Left to Departure", min_value=0, max_value=365, step=1)

    # Make a prediction for user input
    if st.button("Predict Price"):
        user_input = pd.DataFrame({'duration': [duration], 'days_left': [days_left]})
        predicted_price = model.predict(user_input)[0]
        st.success(f"Predicted Price: ${predicted_price:.2f}")

    # Display evaluation metric
    st.header("Model Evaluation")
    st.write("RMSE:", rmse)

    # Visualization Section
    st.header("Data Visualizations")
