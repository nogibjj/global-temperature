# global-temperature
In this project, I want to use Dask parallelization to ingest and process data on global temperature trends and build a command line tool that would make predictions on the temperature of a major city.

![Github Action](https://github.com/nogibjj/global-temperature/actions/workflows/main.yml/badge.svg?event=push)


#### Diagram of this project


#### How the Click CLI works

The main.py file is a click command line tool that allows you to input a city, a country and any year into the future and receive the predicted average yearly annual temperature in that year. 

After you make the python file executable by running the following command:
```
chmod +x main.py
```

You can then run the ./main.py command using the following sequence

```
./main.py --city 'Peking' --country 'China' --year 2040
```

which will prompty return you its prediction: The predicted average temperature in Peking, China in 2040 is 12.7 degrees Celsius.