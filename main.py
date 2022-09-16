#!/usr/bin/env python

"""Building a click command line interface that 
takes in a city and country and returns the linear regression 
of the average temperature of the city on the year"""


# import the dask libraries
import numpy as np
import click
import dask.dataframe as dd
import glob
from dask.array import from_array
from dask_ml.linear_model import LinearRegression

# Create a function that loads in the MajorCity Data and Cleans it
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


# Create a function that takes a City and Country as input and queries the  City Data
def query_city_data(city, country, ddf):
    """Query the  City Data and return mean temperature of the city in each year"""
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
@click.option(
    "--fahrenheit", default=False, help="Return the temperature in Fahrenheit"
)
def main(city, country, year, fahrenheit):
    """Main function that runs the click command"""
    # load the data
    ddf = load_city_data()
    # query the data
    df_city = query_city_data(city, country, ddf)
    # run a linear regression
    lr = linear_regression(df_city)
    # construct a numpy array of the year
    year_array = from_array(np.array([year]).reshape(-1, 1))
    # predict the temperature for the year
    if fahrenheit:
        temp = lr.predict(year_array) * 9 / 5 + 32
        print(
            f"The predicted average temperature in {city}, {country} in {year} is {temp.compute()[0]:.1f} degrees Fareinheit."
        )
    else:
        temp = lr.predict(year_array)
        # print the temperature
        print(
            f"The predicted average temperature in {city}, {country} in {year} is {temp.compute()[0]:.1f} degrees Celsius."
        )


# Run the cli
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
