import streamlit as st
import pandas as pd
import numpy as np
import glob
import dask.dataframe as dd
from dask.array import from_array
from dask_ml.linear_model import LinearRegression
from main import query_city_data, linear_regression

st.title("Global Temperature Predictions")

st.write("""This project allows users to make predictions on the temperature of 
        major cities in the world using historical climate data. The planet has only warmed further since world leaders sat down in 
        Kyoko and signed the Kyoto Protocol in 1997. One driving force of geopolitics in the 21st century will be climate change and 
        the constellation of consequences it will bring. In particular, the coming climate migration will leave a footprint on
        the earth's cultural, demographical and ecological landscape. It is crucial for us to understand where the earth will warm faster
        and by how much. The project was conceived to further this understanding.""")
        

# Create a function that loads in the MajorCity Data and Cleans it
@st.cache
def load_city_data():
    """Load the City Data and Clean it"""
    # check if the city file is in the current directory
    parquet_path = glob.glob("raw_data/part*.parquet")
    # import the city file using read_parquet
    if len(parquet_path) > 0:
        ddf = dd.read_parquet(parquet_path)
    else:
        raise FileNotFoundError("The parquet file is not in the raw_data directory")
    # drop the rows with missing values
    ddf = ddf.dropna()
    # Turn the dt column into a datetime object
    ddf["dt"] = dd.to_datetime(ddf["dt"])
    # Filter out the data that is not in the 20th and 21st century
    ddf = ddf[ddf["dt"] > "1800-01-01"]
    # Rename the dt column to date
    ddf = ddf.rename(columns={"dt": "date"})
    # Create a new column that is the year
    ddf["year"] = ddf["date"].dt.year
    return ddf


data_load_state = st.text("Loading data...")
data = load_city_data()
data_load_state.text("Done! (using st.cache)")


st.subheader("Raw data")
st.write(data.head())

st.write("""A look at all the cities in the data""")
#get the unique cities in the dataframe and cache the data
@st.cache
def get_cities(data):
    # dropping duplicates
    cities_unique = data.drop_duplicates(subset = ["City", "Country"])
    #extract the city and country columns
    cities_countries = cities_unique["City"] + ", " + cities_unique["Country"]
    cities_countries = cities_countries.reset_index(drop = True).compute()
    cities_unique = cities_unique.rename(columns={"Latitude":"lat", "Longitude":"lon"})
    latlon = cities_unique[["lat", "lon"]].reset_index(drop=True)
    latlon = latlon.compute()
    return latlon, cities_countries
    
latlon, cities_countries = get_cities(data)

# clean the dataframe to get their longitudes and lattitudes
df = pd.DataFrame(np.empty((len(latlon),2)), columns=['lat', 'lon'])
for i in range(len(latlon)):
    if latlon.iloc[i,0][-1] == 'N':
        df.iloc[i,0] = float(latlon.iloc[i,0][:-1])
        pass
    else:
        df.iloc[i,0] = float(latlon.iloc[i,0][:-1]) * -1
    if latlon.iloc[i,1][-1] == 'E':
        df.iloc[i,1] = float(latlon.iloc[i,1][:-1])
        pass
    else:
        df.iloc[i,1] = float(latlon.iloc[i,1][:-1]) * -1
st.map(df)
st.caption("""Each dot on the map displays all the cities in the data. At the point the algorithm doesn't support lattitude and longitudes as entry yet.""")

#make selection tools to give inputs
city_option = st.selectbox("Which city would you like to choose?", options = cities_countries)
year = st.slider("What year would you want forecasted?", 2022, 2300, 2050)
fahrenheit = st.checkbox("Temperature in Fahrenheit")
city, country = city_option.split(", ")

#make the prediction
def predict(city, country, year, fahrenheit):
    """Main function that runs the linear regression"""
    # query the data
    df_city = query_city_data(city, country, data)
    # run a linear regression
    lr = linear_regression(df_city)
    # construct a numpy array of the year
    year_array = from_array(np.array([year]).reshape(-1, 1))
    # predict the temperature for the year
    if fahrenheit:
        temp = lr.predict(year_array) * 9 / 5 + 32
    else:
        temp = lr.predict(year_array)
    return temp

temp = predict("New York", "United States", 2050, True)

st.write("The temperature is", )
# temp = predict(city, country, year, fahrenheit)
# delta_temp = temp - 
