import requests
import pandas as pd
import io
# import pygsheets
import datetime as dt
from datetime import timedelta
# from sqlalchemy import create_engine  
from fredapi import Fred
import json

# Connection to RDS Database
# host=End Point
# port= Port Number
# user=
# passw=
# database= Your database name

# Creating an engine
# mydb = create_engine('mysql+pymysql://' + user + ':' + passw + '@' + host + ':' + str(port) + '/' + database , echo=False)


# ---------------------------------------------------------------------------------------------
api_key = (
    "fca12bdb7cef158c35398db24c39443f"  # This is a personalized key to access FRED
)
fred = Fred(api_key)  # Initialize the connection to the FRED Database

# 10 Year Minus 2 Year Treasure
def TenYrTwoYr_api():
    series_2yr = fred.get_series("DGS2", observation_start="2006-01-03")
    series_10yr = fred.get_series("DGS10", observation_start="2006-01-03")
    series_10yr2yr = fred.get_series("T10Y2Y", observation_start="2006-01-03")
    columns = ["Percent"]
    # Put data into data frame
    df_2yr = pd.DataFrame(series_2yr, columns=columns).rename_axis("Date").reset_index()
    df_2yr.rename(columns={"Percent": "% 2yr Maturity"}, inplace=True)
    # Put data into data frame
    df_10yr = (
    pd.DataFrame(series_10yr, columns=columns).rename_axis("Date").reset_index()
    )
    df_10yr.rename(columns={"Percent": "% 10yr Maturity"}, inplace=True)
    # Combine the DFs and create new 10yr - 2yr column
    df_10yr2yr = pd.merge(df_10yr, df_2yr, on="Date")
    df_10yr2yr["% 10Y2Y"] = (df_10yr2yr["% 10yr Maturity"] - df_10yr2yr["% 2yr Maturity"])
    df_10yr2yr['Date'] = pd.to_datetime(df_10yr2yr['Date']).dt.date
    # df_10yr2yr.to_sql(name='TenYrTwoYr', con=mydb, if_exists = 'replace', index=False)
    return df_10yr2yr

# ------------------------------------------------------------------------------------------------------------------------------


# Data for Unemployment claims
def unemp_1_api():
    series_unemp1 = fred.get_series("ICSA", observation_start="2006-01-01")
    columns = ["Percent"]
    # Put data into data frame
    df_unemp1 = (
        pd.DataFrame(series_unemp1, columns=columns).rename_axis("Date").reset_index()
    )
    df_unemp1.rename(columns={"Percent": "Number"}, inplace=True)
    df_unemp1['Date'] = pd.to_datetime(df_unemp1['Date']).dt.date
    return df_unemp1

#  There was a spike in the plotted graph so could not get the clear idea of datapoints from date 2021-08-01
#  Another Api call is made to fetch the data from date 2021-08-01 and plot it.

def unemp_api():
    series_unemp = fred.get_series("ICSA", observation_start="2021-08-01")
    columns = ["Percent"]
    df_unemp = (pd.DataFrame(series_unemp, columns=columns).rename_axis("Date").reset_index())
    df_unemp.rename(columns={"Percent": "Number"}, inplace=True)
    df_unemp['Date'] = pd.to_datetime(df_unemp['Date']).dt.date
    return df_unemp


# Data for Consumer Sentiment
def consumer_sentiment_api():
    series_cc = fred.get_series("UMCSENT", observation_start="2006-01-01")
    columns = ["Percent"]
    df_cc = pd.DataFrame(series_cc, columns=columns).rename_axis("Date").reset_index()
    df_cc.rename(columns={"Percent": "Confidence Level"}, inplace=True)
    df_cc['Date'] = pd.to_datetime(df_cc['Date']).dt.date
    # df_cc.to_sql(name='TenYrTwoYr', con=mydb, if_exists = 'replace', index=False)
    return df_cc
# -------------------------------------------------------------------------------------------------------------------------------


# Consumer Price Index
# Data for Year over Year Percent Change CPI
def YoYPC_api():   
    series_total = fred.get_series(
        "STICKCPIM159SFRBATL", observation_start="2006-01-01")
    series_no_food_energy = fred.get_series(
        "CORESTICKM159SFRBATL", observation_start="2006-01-01")
    columns = ["Percent Change"]

    # Put data into data frame
    df_total = (pd.DataFrame(series_total, columns=columns).rename_axis("Date").reset_index())
    df_total.rename(columns={"Percent Change": "CPI"}, inplace=True)

    # Put data into data frame
    df_no_food_energy = (pd.DataFrame(series_no_food_energy, columns=columns)
        .rename_axis("Date")
        .reset_index())
    df_no_food_energy.rename(
        columns={"Percent Change": "CPI (No Food and Energy)"}, inplace=True)
    df_YoYPC = pd.merge(df_total, df_no_food_energy, on="Date")
     # df_YoYPC.to_sql(name='YoYPC', con=mydb, if_exists = 'replace', index=False)
    return df_YoYPC


# Data for Producer Price Index by Commodity
def ppi_api():
    ppi_series = fred.get_series("PPIACO", observation_start="2006-01-01")
    columns = ["Percent"]
    # Put data into data frame
    df_ppi = (
        pd.DataFrame(ppi_series, columns=columns).rename_axis("Date").reset_index()
    )
    df_ppi.rename(columns={"Percent": "Price Index"}, inplace=True)
    df_ppi['Date'] = pd.to_datetime(df_ppi['Date']).dt.date
    # df_ppi.to_sql(name='PPI', con=mydb, if_exists = 'replace', index=False)
    return df_ppi


# Data for Employment Cost Index
def eci_api():
    eci_series = fred.get_series("ECIALLCIV", observation_start="2006-01-01")
    columns = ["Percent"]
    # Put data into data frame
    df_eci = (pd.DataFrame(eci_series, columns=columns).rename_axis("Date").reset_index())
    df_eci.rename(columns={"Percent": "Index"}, inplace=True)
    df_eci['Date'] = pd.to_datetime(df_eci['Date']).dt.date
    # df_eci.to_sql(name='ECI', con=mydb, if_exists = 'replace', index=False)
    return df_eci
# End-----------------------------------------------------------------------------------------------------------------------



# Data for Average House Sales Price--------------------------------------------------------------------------------------
def house_price_api():
    series_housePrice = fred.get_series("ASPUS", observation_start="2006-01-01")
    columns = ["Percent"]
    # Put data into data frame
    df_housePrice = (pd.DataFrame(series_housePrice, columns=columns).rename_axis("Date").reset_index())
    df_housePrice.rename(columns={"Percent": "Dollars"}, inplace=True)
    df_housePrice['Date'] = pd.to_datetime(df_housePrice['Date']).dt.date
    # df_housePrice.to_sql(name='ECI', con=mydb, if_exists = 'replace', index=False)
    return df_housePrice

# End---------------------------------------------------------------------------------------------------------------------


# Retail Trade and Food Services
# Data for Millions of Dollars
def MD_seasonal_api():
    MD_seasonal = fred.get_series("RSXFS", observation_start="2006-01-01")
    columns = ["Millions of Dollars"]
    # Put data into data frame
    df_MD_seasonal = (pd.DataFrame(MD_seasonal, columns=columns).rename_axis("Date").reset_index())
    df_MD_seasonal['Date'] = pd.to_datetime(df_MD_seasonal['Date']).dt.date
    # df_MD_seasonal.to_sql(name='MD_seasonal', con=mydb, if_exists = 'replace', index=False)
    return df_MD_seasonal

def MD_unseasonal_api():
    MD_not_seasonal = fred.get_series("RSAFSNA", observation_start="2006-01-01")
    columns = ["Millions of Dollars"]
    # Put data into data frame
    df_MD_not_seasonal = (pd.DataFrame(MD_not_seasonal, columns=columns).rename_axis("Date").reset_index())
    df_MD_not_seasonal['Date'] = pd.to_datetime(df_MD_not_seasonal['Date']).dt.date
    # df_MD_not_seasonal.to_sql(name='MD_unseasonal', con=mydb, if_exists = 'replace', index=False)
    return df_MD_not_seasonal

# Data for Percent Change from Preceding Period
def P_C_api():
    P_C = fred.get_series("MARTSMPCSM44000USS", observation_start="2006-01-01",parse_dates=['datetime'])
    columns = ["% Change"]
    # Put data into data frame
    df_P_C = (pd.DataFrame(P_C, columns=columns).rename_axis("Date").reset_index())
    df_P_C['Date'] = pd.to_datetime(df_P_C['Date']).dt.date
    # df_P_C.to_sql(name='PC_seasonal', con=mydb, if_exists = 'replace', index=False)
    return df_P_C

def P_C_unseasonal_api():
    P_C_NS = fred.get_series("MARTSMPCSM44X72USN", observation_start="2006-01-01",parse_dates=['datetime'])
    columns = ["% Change, Not Seasonal"]
    # Put data into data frame
    df_P_C_NS = (pd.DataFrame(P_C_NS, columns=columns).rename_axis("Date").reset_index())
    df_P_C_NS['Date'] = pd.to_datetime(df_P_C_NS['Date']).dt.date
    # df_P_C_NS.to_sql(name='PC_unseasonal', con=mydb, if_exists = 'replace', index=False)
    return df_P_C_NS
# End------------------------------------------------------------------------------------------------------------------------


# Unmployment Rates for States

# Data for Unemployment Rate(Seasonally Adjusted)
def unempr_s_api():
    headers = {'Content-type': 'application/json'}
    # For each state there is different series id..
    data_texas = json.dumps({"seriesid": ['LASST480000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_texas = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_texas, headers=headers)
    json_data_texas = json.loads(p_texas.text)
    data_texas=json_data_texas['Results']['series'][0]['data']
    df_texas = pd.DataFrame.from_dict(data_texas)
    df_texas=df_texas[['year','periodName','value']]
    df_texas=df_texas.rename(columns={'year':'year','periodName':'month','value':'texas_value'})
    data_ohio = json.dumps({"seriesid": ['LASST390000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_ohio = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_ohio, headers=headers)
    json_data_ohio = json.loads(p_ohio.text)
    data_ohio=json_data_ohio['Results']['series'][0]['data']
    df_ohio = pd.DataFrame.from_dict(data_ohio)
    df_ohio=df_ohio[['year','periodName','value']]
    df_ohio=df_ohio.rename(columns={'year':'year','periodName':'month','value':'ohio_value'})
    data_cali = json.dumps({"seriesid": ['LASST060000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_cali = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_cali, headers=headers)
    json_data_cali = json.loads(p_cali.text)
    data_cali=json_data_cali['Results']['series'][0]['data']
    df_cali = pd.DataFrame.from_dict(data_cali)
    df_cali=df_cali[['year','periodName','value']]
    df_cali=df_cali.rename(columns={'year':'year','periodName':'month','value':'cali_value'})
    data_ny = json.dumps({"seriesid": ['LASST360000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_ny = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_ny, headers=headers)
    json_data_ny = json.loads(p_ny.text)
    data_ny=json_data_ny['Results']['series'][0]['data']
    df_ny = pd.DataFrame.from_dict(data_ny)
    df_ny=df_ny[['year','periodName','value']]
    df_ny=df_ny.rename(columns={'year':'year','periodName':'month','value':'ny_value'})
    data_nj = json.dumps({"seriesid": ['LASST340000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_nj = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_nj, headers=headers)
    json_data_nj = json.loads(p_nj.text)
    data_nj=json_data_nj['Results']['series'][0]['data']
    df_nj= pd.DataFrame.from_dict(data_nj)
    df_nj=df_nj[['year','periodName','value']]
    df_nj=df_nj.rename(columns={'year':'year','periodName':'month','value':'nj_value'})
    data_fl = json.dumps({"seriesid": ['LASST120000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_fl = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_fl, headers=headers)
    json_data_fl = json.loads(p_fl.text)
    data_fl=json_data_fl['Results']['series'][0]['data']
    df_fl= pd.DataFrame.from_dict(data_fl)
    df_fl=df_fl[['year','periodName','value']]
    df_fl=df_fl.rename(columns={'year':'year','periodName':'month','value':'fl_value'})
    data_pn = json.dumps({"seriesid": ['LASST420000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_pn = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_pn, headers=headers)
    json_data_pn = json.loads(p_pn.text)
    data_pn=json_data_pn['Results']['series'][0]['data']
    df_pn= pd.DataFrame.from_dict(data_pn)
    df_pn=df_pn[['year','periodName','value']]
    df_pn=df_pn.rename(columns={'year':'year','periodName':'month','value':'pn_value'})
    df_texas['dates']=pd.to_datetime(df_texas['year'].astype(str)  + df_texas['month'], format='%Y%B')
    df_combined_s=df_texas.merge(df_ohio,on=['year','month']).merge(df_cali,on=['year','month']).merge(df_ny,on=['year','month']).merge(df_nj,on=['year','month']).merge(df_fl,on=['year','month']).merge(df_pn,on=['year','month'])
    df_combined_s=df_combined_s.drop(['year','month'],axis=1)
    first_column = df_combined_s.pop('dates')
    # combined all data in one dataframe
    df_combined_s.insert(0, 'dates', first_column)
    df_combined_s[['texas_value', 'ohio_value', 'cali_value', 'ny_value','nj_value', 'fl_value', 'pn_value']]=df_combined_s[['texas_value', 'ohio_value', 'cali_value', 'ny_value','nj_value', 'fl_value', 'pn_value']].apply(pd.to_numeric)    
    df_combined_s['dates'] = pd.to_datetime(df_combined_s['dates']).dt.date
    return df_combined_s

    

# Data for Unemployment Rate(Not Seasonally Adjusted)
def unempr_us_api():
    headers = {'Content-type': 'application/json'}
    data_texas = json.dumps({"seriesid": ['LAUST480000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_texas = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_texas, headers=headers)
    json_data_texas = json.loads(p_texas.text)
    data_texas=json_data_texas['Results']['series'][0]['data']
    df_texas = pd.DataFrame.from_dict(data_texas)
    df_texas=df_texas[['year','periodName','value']]
    df_texas=df_texas.rename(columns={'year':'year','periodName':'month','value':'texas_value'})
    data_ohio = json.dumps({"seriesid": ['LAUST390000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_ohio = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_ohio, headers=headers)
    json_data_ohio = json.loads(p_ohio.text)
    data_ohio=json_data_ohio['Results']['series'][0]['data']
    df_ohio = pd.DataFrame.from_dict(data_ohio)
    df_ohio=df_ohio[['year','periodName','value']]
    df_ohio=df_ohio.rename(columns={'year':'year','periodName':'month','value':'ohio_value'})
    data_cali = json.dumps({"seriesid": ['LAUST060000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_cali = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_cali, headers=headers)
    json_data_cali = json.loads(p_cali.text)
    data_cali=json_data_cali['Results']['series'][0]['data']
    df_cali = pd.DataFrame.from_dict(data_cali)
    df_cali=df_cali[['year','periodName','value']]
    df_cali=df_cali.rename(columns={'year':'year','periodName':'month','value':'cali_value'})
    data_ny = json.dumps({"seriesid": ['LAUST360000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_ny = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_ny, headers=headers)
    json_data_ny = json.loads(p_ny.text)
    data_ny=json_data_ny['Results']['series'][0]['data']
    df_ny = pd.DataFrame.from_dict(data_ny)
    df_ny=df_ny[['year','periodName','value']]
    df_ny=df_ny.rename(columns={'year':'year','periodName':'month','value':'ny_value'})
    data_nj = json.dumps({"seriesid": ['LAUST340000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_nj = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_nj, headers=headers)
    json_data_nj = json.loads(p_nj.text)
    data_nj=json_data_nj['Results']['series'][0]['data']
    df_nj= pd.DataFrame.from_dict(data_nj)
    df_nj=df_nj[['year','periodName','value']]
    df_nj=df_nj.rename(columns={'year':'year','periodName':'month','value':'nj_value'})
    data_fl = json.dumps({"seriesid": ['LAUST120000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_fl = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_fl, headers=headers)
    json_data_fl = json.loads(p_fl.text)
    data_fl=json_data_fl['Results']['series'][0]['data']
    df_fl= pd.DataFrame.from_dict(data_fl)
    df_fl=df_fl[['year','periodName','value']]
    df_fl=df_fl.rename(columns={'year':'year','periodName':'month','value':'fl_value'})
    data_pn = json.dumps({"seriesid": ['LAUST420000000000003'],"startyear":"2014","endyear":"2022","registrationkey":"05264e50d6ad48b9a296495c5db0a2b2"})
    p_pn = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data_pn, headers=headers)
    json_data_pn = json.loads(p_pn.text)
    data_pn=json_data_pn['Results']['series'][0]['data']
    df_pn= pd.DataFrame.from_dict(data_pn)
    df_pn=df_pn[['year','periodName','value']]
    df_pn=df_pn.rename(columns={'year':'year','periodName':'month','value':'pn_value'})
    df_texas['dates']=pd.to_datetime(df_texas['year'].astype(str)  + df_texas['month'], format='%Y%B')
    df_combined_us=df_texas.merge(df_ohio,on=['year','month']).merge(df_cali,on=['year','month']).merge(df_ny,on=['year','month']).merge(df_nj,on=['year','month']).merge(df_fl,on=['year','month']).merge(df_pn,on=['year','month'])
    df_combined_us=df_combined_us.drop(['year','month'],axis=1)
    first_column = df_combined_us.pop('dates')
    df_combined_us.insert(0, 'dates', first_column)
    df_combined_us[['texas_value', 'ohio_value', 'cali_value', 'ny_value','nj_value', 'fl_value', 'pn_value']]=df_combined_us[['texas_value', 'ohio_value', 'cali_value', 'ny_value','nj_value', 'fl_value', 'pn_value']].apply(pd.to_numeric)    
    df_combined_us['dates'] = pd.to_datetime(df_combined_us['dates']).dt.date
    # df_combined_us.to_sql(name='PC_unseasonal', con=mydb, if_exists = 'replace', index=False)
    return df_combined_us

# End----------------------------------------------------------------------------------------



# ---------------------------------- Real GDP Growth--------------------------------


def Real_GDP_api():
    gdp_data = fred.get_series("GDPC1", observation_start="1999-01-01")
    columns=["GDP"]
    gdp_df = pd.DataFrame(gdp_data, columns=columns).rename_axis("Date").reset_index()
    gdp_df['Annualized QoQ Real GDP Growth'] = 4*(gdp_df.GDP.div(gdp_df.GDP.shift()).fillna(gdp_df.GDP)-1)
    gdp_df['YoY Real GDP Growth'] = gdp_df.GDP.div(gdp_df.GDP.shift(4)).fillna(gdp_df.GDP)-1
    gdp_df=gdp_df.iloc[4:,::]
    return gdp_df

# End--------------------------------------------------------------------------------





















