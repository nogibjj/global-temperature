"""Building a click command line interface that 
takes in a city and country and returns the linear regression 
of the average temperature of the city on the year"""


# import the dask libraries
from dask.array import from_array
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client
from dask_ml.linear_model import LinearRegression
import numpy as np
import click

# Create a function that loads in the MajorCity Data and Cleans it
def load_major_city_data():
    """Load the Major City Data and Clean it"""
    # Load the data into a dask dataframe
    ddf = dd.read_csv("raw_data/GlobalLandTemperaturesByMajorCity.csv")
    # Turn the dt column into a datetime object
    ddf["dt"] = dd.to_datetime(ddf["dt"])
    # Filter out the data that is not in the 20th and 21st century
    ddf = ddf[(ddf["dt"] > "1900-01-01") & (ddf["dt"] < "2100-01-01")]
    # Rename the dt column to date
    ddf = ddf.rename(columns={"dt": "date"})
    # Create a new column that is the year
    ddf["year"] = ddf["date"].dt.year
    return ddf


# Create a function that takes a City and Country as input and queries the Major City Data
def query_major_city_data(city, country, ddf):
    """Query the Major City Data and return mean temperature of the city in each year"""
    # Filter the data to the city and country
    df_city = ddf[(ddf["City"] == city) & (ddf["Country"] == country)]
    # Group the data by year and calculate the mean temperature
    df_city = df_city.groupby("year").mean().reset_index()
    # Return the dataframe
    return df_city


# Create a function that runs a linear regression of the average temperature of the city on the year
def linear_regression(ddf):
    """Run a linear regression on the data"""
    # Create a linear regression object
    lr = LinearRegression()
    # Fit the linear regression model
    lr.fit(ddf[["year"]].to_dask_array(), ddf[["AverageTemperature"]].to_dask_array())
    # Return the linear regression object
    return lr


# Create a click command that takes in city, country and year as input and returns the predicted average temperature
@click.command()
@click.option("--city", default="London", help="The city to analyze")
@click.option("--country", default="United Kingdom", help="The country to analyze")
@click.option("--year", default=2010, help="The year to predict")
def main(city, country, year):
    """Predict the average temperature of a city in a given year"""
    # create a local cluster in a context manager
    with LocalCluster() as cluster, Client(cluster) as client:
        # load the data
        ddf = load_major_city_data()
        # query the data
        df_city = query_major_city_data(city, country, ddf)
        # run a linear regression
        lr = linear_regression(df_city)
        # construct a numpy array of the year
        year_array = from_array(np.array([year]).reshape(-1, 1))
        # predict the temperature for the year
        temp = lr.predict(year_array).compute()
        # print the temperature
        print(
            f"The predicted average temperature in {city}, {country} in {year} is {temp[0]:.1f} degrees Celsius"
        )


# Run the cli
if __name__ == "__main__":
    main()
