# global-temperature
In this project, I want to use Dask parallelization to ingest and process data on global temperature trends and build a command line tool that would make predictions on the temperature of a major city.

![Github Action](https://github.com/nogibjj/global-temperature/actions/workflows/main.yml/badge.svg?event=push)


#### Diagram of this project

![Project Diagram](https://user-images.githubusercontent.com/60377132/190940809-9189d7fa-6b97-484a-b2a4-eff0e302e8da.jpg)

#### How the Click CLI works

The main.py file is a click command line tool that allows you to input a city, a country and any year into the future and receive the predicted average yearly annual temperature in that year. 

After you make the python file executable by running the following command:
```
chmod +x main.py
```

You can then run the ./main.py command using the following sequence

```
./main.py --city 'Peking' --country 'China' --year 2040 -- fahrenheit false
```

which will prompty return you its prediction: The predicted average temperature in Peking, China in 2040 is 12.7 degrees Celsius. Entering true for the fahrenheit option would return the temperature in Fahrenheit.

#### How the python script works behind the scene

The script is run on around temperature data of 3,500 cities in the world. The data is stored in parquet format in the raw_data directory and contains average yearly temperatures of cities going back to 1700s. 

The python script loads in the data using Dask, a parallel computing program, and the exploratory data analysis and its plots are stored in the eda.ipynb file. The script loads the data in, cleans the NA values and select only the yearly temperature from 1800 onward, and quries for a city by its city and country name. The program then runs a linear regression on the temperature data using historical values. It then makes a prediction using the year inputed by the user.  