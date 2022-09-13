# import the dask libraries
import dask
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client
from dask_ml.linear_model import LinearRegression
import click



# Create a function that loads in the MajorCity Data and Cleans it
def load_major_city_data():
    """Load the Major City Data and Clean it"""
    # Load the data into a dask dataframe
    df = dd.read_csv('raw_data/GlobalLandTemperaturesByMajorCity.csv')
    # Turn the dt column into a datetime object
    df['dt'] = dd.to_datetime(df['dt'])
    # Filter out the data that is not in the 20th and 21st century
    df = df[(df['dt'] > '1900-01-01') & (df['dt'] < '2100-01-01')]
    # Rename the dt column to date
    df = df.rename(columns={'dt': 'date'})
    # Create a new column that is the year
    df['year'] = df['date'].dt.year
    return df

# Create a function that takes a City and Country as input and queries the Major City Data
def query_major_city_data(city, country, df):
    """Query the Major City Data and return mean temperature of the city in each year"""
    # Filter the data to the city and country
    df_city = df[(df['City'] == city) & (df['Country'] == country)]
    # Group the data by year and calculate the mean temperature
    df_city = df_city.groupby('year').mean()
    # Return the dataframe
    return df_city

# Create a function that runs a linear regression of the average temperature of the city on the year
def linear_regression(df):
    """Run a linear regression on the data"""
    # Create a linear regression object
    lr = LinearRegression()
    # Fit the linear regression model
    lr.fit(df['year'].to_frame(), df['AverageTemperature'])
    # Return the linear regression object
    return lr



# start the cluster
cluster = LocalCluster()
client = Client(cluster)

# Create a function that subset the data 