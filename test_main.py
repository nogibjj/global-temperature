import dask.dataframe as dd
import main

# test that the data will load
def test_load_major_city_data():
    """Test that the data will load"""
    ddf = main.load_city_data()
    assert isinstance(ddf, dd.DataFrame)
    assert ddf.shape[0].compute() > 10000
    assert ddf.shape[1] == 8
    assert "year" in ddf.columns


# test that the data will query
def test_query_major_city_data():
    """Test that the data will query"""
    ddf = main.load_city_data()
    df_city = main.query_city_data("London", "United Kingdom", ddf)
    assert isinstance(df_city, dd.DataFrame)
    assert df_city.shape[0].compute() > 100
    assert df_city.shape[1] == 3
    assert "year" in df_city.columns
    assert "AverageTemperature" in df_city.columns


# test that the linear regression will run
def test_linear_regression():
    """Test that the linear regression will run"""
    ddf = main.load_city_data()
    df_city = main.query_city_data("London", "United Kingdom", ddf)
    lr = main.linear_regression(df_city)
    assert hasattr(lr, "predict")
