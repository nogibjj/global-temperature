import streamlit as st
import pandas as pd
import numpy as np
import glob
import dask.dataframe as dd
from sklearn.linear_model import LinearRegression

st.title("Global Temperature Predictions")

st.write(
    """This project allows users to make predictions on the temperature of 
        major cities in the world using historical climate data. The planet has only warmed further since world leaders sat down in 
        Kyoko and signed the Kyoto Protocol in 1997. One driving force of geopolitics in the 21st century will be climate change and 
        the constellation of consequences it will bring. In particular, the coming climate migration will leave a footprint on
        the earth's cultural, demographical and ecological landscape. It is crucial for us to understand where the earth will warm faster
        and by how much. The project was conceived to further this understanding."""
)


# Create a function that loads in the MajorCity Data and Cleans it
@st.cache
def load_city_data():
    """Load the City Data and Clean it"""
    # check if the city file is in the current directory
    parquet_path = glob.glob("*.parquet")
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
df = load_city_data()
data_load_state.text("Done! (using st.cache)")


st.subheader("Raw data")
st.write(df.head())

st.subheader("""A look at all the cities in the data""")
# get the unique cities in the dataframe and cache the data
@st.cache
def get_cities(data_frame):
    # dropping duplicates
    cities_unique = data_frame.drop_duplicates(subset=["City", "Country"])
    # extract the city and country columns
    cities = cities_unique["City"] + ", " + cities_unique["Country"]
    cities = cities.reset_index(drop=True).compute()
    cities_unique = cities_unique.rename(
        columns={"Latitude": "lat", "Longitude": "lon"}
    )
    lat_lon = cities_unique[["lat", "lon"]].reset_index(drop=True).compute()
    return lat_lon, cities


latlon, cities_countries = get_cities(df)

# clean the dataframe to get their longitudes and lattitudes
frame = pd.DataFrame(np.empty((len(latlon), 2)), columns=["lat", "lon"])
for i in range(len(latlon)):
    if latlon.iloc[i, 0][-1] == "N":
        frame.iloc[i, 0] = float(latlon.iloc[i, 0][:-1])
    else:
        frame.iloc[i, 0] = float(latlon.iloc[i, 0][:-1]) * -1
    if latlon.iloc[i, 1][-1] == "E":
        frame.iloc[i, 1] = float(latlon.iloc[i, 1][:-1])
    else:
        frame.iloc[i, 1] = float(latlon.iloc[i, 1][:-1]) * -1
st.map(frame)
st.caption(
    """Each dot on the map displays all the cities in the data. The algorithm doesn't support lattitude and longitudes as entry yet."""
)

st.subheader("Make your predictions")
st.write(
    """Toggle through the following cities and select a year to see the prediction for the average temperature for that year.
            You can choose to display the temperature in Fahrenheit or Celsius."""
)

# make selection tools to give inputs

col1, col2 = st.columns(2)
with col1:
    city_option = st.selectbox(
        "Which city would you like to choose?", options=cities_countries
    )
    fahrenheit = st.checkbox("Temperature in Fahrenheit")
with col2:
    year = st.slider("What year would you want forecasted?", 2022, 2300, 2050)

City, Country = city_option.split(", ")


def query_city_data(city, country, ddf):
    """Query the City Data and return mean temperature of the city in each year"""
    # Filter the data to the city and country
    city_dataframe = ddf[(ddf["City"] == city) & (ddf["Country"] == country)]
    # Group the data by year and calculate the mean temperature
    city_dataframe = city_dataframe.groupby("year").mean().reset_index().compute()
    # Return the dataframe
    return city_dataframe


# make the prediction
def predict(city, country, Year, Fahrenheit):
    """Main function that runs the linear regression"""
    # query the data
    df_city = query_city_data(city, country, df)
    # run a linear regression
    lr = LinearRegression()
    lr.fit(df_city[["year"]], df_city[["AverageTemperature"]])
    # construct a numpy array of the year
    year_array = np.array([[Year]])
    # predict the temperature for the year
    if Fahrenheit:
        temp = lr.predict(year_array) * 9 / 5 + 32
    else:
        temp = lr.predict(year_array)
    return temp, df_city


temperature, city_data = predict(City, Country, year, fahrenheit)
temp_1800 = city_data.iloc[0, 1]
df_new = city_data[["year", "AverageTemperature"]].set_index("year")

col1, col2 = st.columns(2)
with col1:
    if fahrenheit:
        temp_1800 = temp_1800 * 9 / 5 + 32
        st.metric(
            "Forecasted Average Temperature",
            f"{int(temperature):.2f}°F",
            f"{int(temperature - temp_1800):.2f}°F",
        )
    else:
        st.metric(
            "Forecasted Average Temperature",
            f"{int(temperature):.2f}°C",
            f"{int(temperature - temp_1800):.2f}°C",
        )
    st.caption(
        "The delta value signifies the change in annual temperature since year 1800."
    )

with col2:
    if fahrenheit:
        df_new[["AverageTemperature"]] = df_new[["AverageTemperature"]] * 9 / 5 + 32
    st.line_chart(df_new)
