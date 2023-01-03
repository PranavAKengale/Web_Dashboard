from markdown import markdown
import pandas as pd
from datetime import datetime
import streamlit as st
import datetime as dt
import plotly.offline as py
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import numpy as np


st.set_page_config(
layout = 'wide')
st.markdown("<h1 style='text-align: center; color: White;'>Biz2X Macroeconomic Dashboard</h1>", unsafe_allow_html=True) 

pd.set_option('display.max_colwidth', None)
selected=option_menu(
    menu_title=None,
    options=["10 Year Minus 2 Year Treasury","Unemployment Claims","Consumer Sentiment","Price Index","Average House Sales Price",
                    "Retail Trade and Food Services","Unemployment Rates for States","Real GDP Growth","Payroll Benchmarks","Payroll Benchmarks (Sub Industry)"],
    default_index=7,
    orientation="horizontal")




#---------------------------------------------------  10 Year 2 Year graph-----------------------------------------------------------------------
def TenYrTwoYr():
    from api_calls import TenYrTwoYr_api
    df_10yr2yr=TenYrTwoYr_api()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_10yr2yr.Date,
            y=df_10yr2yr["% 10yr Maturity"],
            name="10 Year Maturity",
            line_color="#BAB0AC",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_10yr2yr.Date,
            y=df_10yr2yr["% 2yr Maturity"],
            name="2 Year Maturity",
            line_color="#0D2A63",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_10yr2yr.Date,
            y=df_10yr2yr["% 10Y2Y"],
            name="10 Year Minus 2 Year",
            line_color="#16FF32",
        )
    )

    # Formatting
    fig.update_layout(width=1400,height=600)
    fig.update_layout(xaxis_title="Date", yaxis_title="Percent")
    fig.update_layout(template="plotly_dark")

    fig.update_yaxes(tick0=-1.5, dtick=0.5)
    fig.update_layout(hovermode="x unified")
    # st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
    df_10yr2yr['Date'] = pd.to_datetime(df_10yr2yr['Date']).dt.date
    df_10yr2yr = df_10yr2yr.iloc[::-1]
    df_10yr2yr = df_10yr2yr.reset_index(drop=True)
    # st.dataframe(df_10yr2yr[["% 10yr Maturity","% 2yr Maturity","% 10Y2Y"]].style.format("{:.2%}"))
    # df_10yr2yr[["% 10yr Maturity","% 2yr Maturity","% 10Y2Y"]] = df_10yr2yr[["% 10yr Maturity","% 2yr Maturity","% 10Y2Y"]].round(decimals=2)

    
    st.markdown("<h2 style='text-align: left; color: #16FF32;'>10 Year Treasury Minus 2 Year Treasury</h2>", 
    unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, sharing="streamlit")
 
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_10yr2yr)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='10Y2Y.csv',
        mime='text/csv')
    placeholder = st.empty()
    if st.checkbox("Show Dataframe"):
        st.dataframe(df_10yr2yr.style.format({'% 10yr Maturity': '{:.2f}', '% 2yr Maturity': '{:.2f}', '% 10Y2Y': '{:.2f}'}),use_container_width=True)
# End-------------------------------------------------------------------------------------------------


# Unemployment Claims----------------------------------------------------------------------------------
def unemployment():
    from api_calls import unemp_1_api
    df_unemp1=unemp_1_api()

    unemp_1_fig = go.Figure()
    unemp_1_fig.add_trace(
        go.Scatter(
            x=df_unemp1.Date,
            y=df_unemp1["Number"],
            name="Unemp Claims",
            line_color="#16FF32",
        )
    )

    unemp_1_fig.update_layout(xaxis_title="Date", yaxis_title="Number", xaxis_rangeselector_font_color='black',
                    xaxis_rangeselector_activecolor='#16FF32',
                    xaxis_rangeselector_bgcolor='white')
    unemp_1_fig.update_layout(template="none")
    unemp_1_fig.update_layout(hovermode="x unified")
    unemp_1_fig.update_yaxes(tick0=0, dtick=2000000)
    unemp_1_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
        
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                    
                ]
            )
        ),
    )

    from api_calls import unemp_api
    df_unemp= unemp_api()

    unemp_fig = go.Figure()

    unemp_fig.add_trace(
        go.Scatter(
            x=df_unemp.Date,
            y=df_unemp["Number"],
            name="Unemp Claims",
            line_color="#16FF32",
        )
    )

    unemp_fig.update_layout(xaxis_title="Date", yaxis_title="Number", xaxis_rangeselector_font_color='black',
                    xaxis_rangeselector_activecolor='#16FF32',
                    xaxis_rangeselector_bgcolor='white')
    unemp_fig.update_layout(template="none")
    unemp_fig.update_layout(hovermode="x unified")
    unemp_fig.update_yaxes(tick0=0, dtick=40000)


    unemp_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
        
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                    
                ]
            )
        ),

    )

    df_unemp1['Date'] = pd.to_datetime(df_unemp1['Date']).dt.date
    df_unemp1['Number']=df_unemp1['Number'].astype(int)
    df_unemp1 = df_unemp1.iloc[::-1]
    df_unemp1 = df_unemp1.reset_index(drop=True)
    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Initial Unemployment Claims</h2>", 
    unsafe_allow_html=True)
    st.plotly_chart(unemp_1_fig, use_container_width=True, sharing="streamlit")
    st.plotly_chart(unemp_fig, use_container_width=True, sharing="streamlit")

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_unemp1)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Unemployment Claims.csv',
        mime='text/csv',
 )
    placeholder = st.empty()
    if st.checkbox("Show Dataframe"):
        st.dataframe(df_unemp1,use_container_width=True)

#  END-----------------------------------------------------------------------------------------------

# Consumer Sentiment----------------------------------------------------------------------------------

def consumer_sentiment():
    from api_calls import consumer_sentiment_api
    df_cc=consumer_sentiment_api()
    threshold_high = 81
    threshold_low = 71

    fig4 = go.Figure()

    fig4.add_trace(     
        go.Scatter(
            x=df_cc.Date,
            y=df_cc["Confidence Level"],
            name="Average",
            line_color="#BAB0AC",
        )
    )
    fig4.add_trace(
        go.Scattergl(
            x=df_cc.Date,
            y=df_cc["Confidence Level"].where(
                df_cc["Confidence Level"] < threshold_low
            ),
            name="Below Average",
            line_color=("#0D2A63"),
        )
    )

    fig4.add_trace(
        go.Scattergl(
            x=df_cc.Date,
            y=df_cc["Confidence Level"].where(
                df_cc["Confidence Level"] > threshold_high
            ),
            line={"color": "#16FF32"},
            name="Above Average",
        )
    )

    fig4.update_layout(xaxis_title="Date", yaxis_title="Consumer Sentiment Level")
    fig4.update_layout(template="none")
    fig4.update_yaxes(range=[45, 105])
    fig4.update_layout(hovermode="x unified")
    fig4.update_layout(
        annotations=[
            go.layout.Annotation(
                text="Average Confidence: 71-81",
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=1.10,
                y=1.08,
                bordercolor="white",
                borderwidth=1,
            )
        ]
    )

    df_cc['Date'] = pd.to_datetime(df_cc['Date']).dt.date
    df_cc = df_cc.iloc[::-1]
    df_cc = df_cc.reset_index(drop=True)

    st.markdown("<h2 style='text-align: left; color: #16FF32;'>University of Michigan Consumer Sentiment</h2>",
    unsafe_allow_html=True)
    st.plotly_chart(fig4, use_container_width=True, sharing="streamlit")

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_cc)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Consumer Sentiment.csv',
        mime='text/csv',
 )
    placeholder = st.empty()
    if st.checkbox("Show Dataframe"):
         st.dataframe(df_cc.style.format({"Confidence Level": '{:.1f}'}),use_container_width=True)

#  END---------------------------------------------------------------------------------------------


# Consumer Price Index------------------------------------------------------------------------------
def CPI():
    options=["Select an index","Year over Year Percent Change CPI","Producer Price Index by Commodity: All Commodities",
    "Employment Cost Index: Total compensation: All Civilian"]
    selection=st.selectbox('',options=options)
    return selection

def YoYPC():   

    from api_calls import YoYPC_api
    df_YoYPC=YoYPC_api()
    df_YoYPC['Date'] = pd.to_datetime(df_YoYPC['Date']).dt.date
    YoYPC_fig = go.Figure()
    YoYPC_fig.add_trace(
        go.Scatter(
            x=df_YoYPC.Date, y=df_YoYPC["CPI"], name="CPI", line_color="#CE2029",
        )
    )

    YoYPC_fig.add_trace(
        go.Scatter(
            x=df_YoYPC.Date,
            y=df_YoYPC["CPI (No Food and Energy)"],
            name="CPI less Food and Energy",
            line_color="#16FF32",
        )
    )

    #YoYPC_fig.update_layout(title_text="Year over Year Percent Change CPI")
    YoYPC_fig.update_layout(xaxis_title="Date", yaxis_title="Percent Change from Year Ago")
    YoYPC_fig.update_layout(template="plotly_dark")

    YoYPC_fig.update_yaxes(tick0=-1.5, dtick=0.5)
    YoYPC_fig.update_layout(hovermode="x unified")

    df_YoYPC['Date'] = pd.to_datetime(df_YoYPC['Date']).dt.date
    df_YoYPC = df_YoYPC.iloc[::-1]
    df_YoYPC = df_YoYPC.reset_index(drop=True)
    
    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Year over Year Percent Change CPI</h2>", 
    unsafe_allow_html=True)
    st.plotly_chart(YoYPC_fig, use_container_width=True, sharing="streamlit")

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_YoYPC)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Consumer Price Index.csv',
        mime='text/csv')
    placeholder = st.empty()
    if st.checkbox("Show Dataframe"):
        st.dataframe(df_YoYPC.style.format({"CPI": '{:.2f}', "CPI (No Food and Energy)": '{:.2f}'}),use_container_width=True)


def ppi():
    from api_calls import ppi_api
    df_ppi=ppi_api()
    ppi_fig = go.Figure()
    ppi_fig.add_trace(
        go.Scatter(
            x=df_ppi.Date,
            y=df_ppi["Price Index"],
            #name="Unemp Claims",
            line_color="#16FF32"))

    ppi_fig.update_layout(xaxis_title="Date", yaxis_title="Price Index (1982 = 100)", xaxis_rangeselector_font_color='black',
                    xaxis_rangeselector_activecolor='#16FF32',
                    xaxis_rangeselector_bgcolor='white')
    ppi_fig.update_layout(template="none")
    ppi_fig.update_layout(hovermode="x unified")
    #ppi_fig.update_yaxes(tick0=0, dtick=2000000)
    ppi_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
        
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                    
                ]
            )
        ),
    )

    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Producer Price Index: All Commodities</h2>",
    unsafe_allow_html=True)
    st.plotly_chart(ppi_fig, use_container_width=True, sharing="streamlit")

    df_ppi['Date'] = pd.to_datetime(df_ppi['Date']).dt.date
    df_ppi = df_ppi.iloc[::-1]
    df_ppi = df_ppi.reset_index(drop=True)

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_ppi)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='PPI.csv',
        mime='text/csv',
)
    placeholder = st.empty()
    if st.checkbox("Show Dataframe"):
        st.dataframe(df_ppi.style.format({"Price Index": '{:.2f}'}),use_container_width=True)
# End--------------------------------------------------------------------------------------------



# ECI ------------------------------------------------------------------------------------------------

def eci():
    from api_calls import eci_api
    df_eci=eci_api()
    eci_fig = go.Figure()

    eci_fig.add_trace(
        go.Scatter(
            x=df_eci.Date,
            y=df_eci["Index"],
            #name="Unemp Claims",
            line_color="#16FF32",
        )
    )

    eci_fig.update_layout(xaxis_title="Date", yaxis_title="Index (1982 = 100)", xaxis_rangeselector_font_color='black',
                    xaxis_rangeselector_activecolor='#16FF32',
                    xaxis_rangeselector_bgcolor='white')
    eci_fig.update_layout(template="none")
    eci_fig.update_layout(hovermode="x unified")
    #ppi_fig.update_yaxes(tick0=0, dtick=2000000)
    eci_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
        
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                    
                ]
            )
        ),
    )

    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Employment Cost Index: Total compensation: All Civilian</h2>",
    unsafe_allow_html=True)
    st.plotly_chart(eci_fig, use_container_width=True, sharing="streamlit")

    df_eci['Date'] = pd.to_datetime(df_eci['Date']).dt.date
    df_eci = df_eci.iloc[::-1]
    df_eci = df_eci.reset_index(drop=True)

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_eci)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='ECI.csv',
        mime='text/csv',
)
    placeholder = st.empty()
    if st.checkbox("Show Dataframe"):
        st.dataframe(df_eci.style.format({"Index": '{:.1f}'}),use_container_width=True)


# Average House Sales Price-----------------------------------------------------------------------

def house_price():
    from api_calls import house_price_api
    df_housePrice=house_price_api()
    housePrice_fig = go.Figure()
    housePrice_fig.add_trace(
        go.Scatter(
            x=df_housePrice.Date,
            y=df_housePrice["Dollars"],
            name="Average House Price",
            line_color="#16FF32",
        )
    )

    housePrice_fig.update_layout(xaxis_title="Date", yaxis_title="Dollars", xaxis_rangeselector_font_color='black',
                    xaxis_rangeselector_activecolor='#16FF32',
                    xaxis_rangeselector_bgcolor='white')
    housePrice_fig.update_layout(template="none")
    housePrice_fig.update_layout(hovermode="x unified")
    housePrice_fig.update_yaxes(tick0=0, dtick=40000)


    housePrice_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
        
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                    
                ]
            )
        ),

    )

    df_housePrice['Date'] = pd.to_datetime(df_housePrice['Date']).dt.date
    df_housePrice = df_housePrice.iloc[::-1]
    df_housePrice = df_housePrice.reset_index(drop=True)

    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Average Sales Price of Houses Sold (U.S.A. Quarterly)</h2>", 
    unsafe_allow_html=True)
    st.plotly_chart(housePrice_fig, use_container_width=True, sharing="streamlit")
 
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_housePrice)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Average House Sales Price.csv',
        mime='text/csv',
 )
    placeholder = st.empty()
    if st.checkbox("Show Dataframe"):
        st.dataframe(df_housePrice.style.format({"Dollars": '{:.0f}'}),use_container_width=True)

# End----------------------------------------------------------------------------------------------------


# Retail Trade & Food Services---------------------------------------------------------------------------
def Retail():
    options=["Select an index","Millions of Dollars","Percent Change from Preceding Period"]
    selection=st.selectbox('',options=options)
    return selection


def MD():
    from api_calls import MD_seasonal_api
    df_MD_seasonal=MD_seasonal_api()
    MD_seasonal_fig = go.Figure()
    MD_seasonal_fig.add_trace(
        go.Scatter(
            x=df_MD_seasonal.Date,
            y=df_MD_seasonal["Millions of Dollars"],
            line_color="#16FF32",
        )
    )

    MD_seasonal_fig.update_layout(xaxis_title="Date", yaxis_title="Millions of Dollars", xaxis_rangeselector_font_color='black',
                    xaxis_rangeselector_activecolor='#16FF32',
                    xaxis_rangeselector_bgcolor='white')
    MD_seasonal_fig.update_layout(template="none")
    MD_seasonal_fig.update_layout(hovermode="x unified")
    MD_seasonal_fig.update_yaxes(tick0=0, dtick=2000000)
    MD_seasonal_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(

            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),

                ]
            )
        ),
    )
    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Millions of Dollars, Seasonally Adjusted</h2>", 
    unsafe_allow_html=True)
    st.plotly_chart(MD_seasonal_fig, use_container_width=True, sharing="streamlit")

    df_MD_seasonal['Date'] = pd.to_datetime(df_MD_seasonal['Date']).dt.date
    df_MD_seasonal = df_MD_seasonal.iloc[::-1]
    df_MD_seasonal = df_MD_seasonal.reset_index(drop=True)

    def convert_df_seasonal(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df_seasonal(df_MD_seasonal)
    st.download_button(
        label="Download data as CSV (Seasonally Adjusted)",
        data=csv,
        file_name='MD_Seasonally_Adjusted.csv',
        mime='text/csv')
    placeholder = st.empty()
    if st.checkbox("Show Dataframe (Seasonally Adjusted)"):
        st.dataframe(df_MD_seasonal.style.format({"Millions of Dollars": '{:.0f}'}),use_container_width=True)


#  For not-seasonally adjusted data
    from api_calls import MD_unseasonal_api
    df_MD_not_seasonal=MD_unseasonal_api()
    MD_not_seasonal_fig = go.Figure()
    MD_not_seasonal_fig = go.Figure()

    MD_not_seasonal_fig.add_trace(
        go.Scatter(
            x=df_MD_not_seasonal.Date,
            y=df_MD_not_seasonal["Millions of Dollars"],
            line_color="#16FF32",
        )
    )
    MD_not_seasonal_fig.update_layout(xaxis_title="Date", yaxis_title="Millions of Dollars", xaxis_rangeselector_font_color='black',
                    xaxis_rangeselector_activecolor='#16FF32',
                    xaxis_rangeselector_bgcolor='white')
    MD_not_seasonal_fig.update_layout(template="none")
    MD_not_seasonal_fig.update_layout(hovermode="x unified")
    MD_not_seasonal_fig.update_yaxes(tick0=0, dtick=2000000)
    MD_not_seasonal_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(

            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),

                ]
            )
        ),
    )
    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Millions of Dollars, Not Seasonally Adjusted</h2>", 
    unsafe_allow_html=True)
    st.plotly_chart(MD_not_seasonal_fig, use_container_width=True, sharing="streamlit")

    df_MD_not_seasonal['Date'] = pd.to_datetime(df_MD_not_seasonal['Date']).dt.date
    df_MD_not_seasonal = df_MD_not_seasonal.iloc[::-1]
    df_MD_not_seasonal = df_MD_not_seasonal.reset_index(drop=True)   

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_MD_not_seasonal)
    st.download_button(
        label="Download data as CSV (Not Seasonally Adjusted)",
        data=csv,
        file_name='MD_Seasonally_Adjusted.csv',
        mime='text/csv',
)
    placeholder = st.empty()
    if st.checkbox("Show Dataframe (Not Seasonally Adjusted)"):
        st.dataframe(df_MD_not_seasonal.style.format({"Millions of Dollars": '{:.0f}'}),use_container_width=True)

#End----------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
def P_C():
    from api_calls import P_C_api
    df_P_C=P_C_api()
    P_C_fig = go.Figure()
    P_C_fig = go.Figure()

    P_C_fig.add_trace(
        go.Scatter(
            x=df_P_C.Date,
            y=df_P_C["% Change"],
            line_color="#16FF32",
        )
    )

    P_C_fig.update_layout(xaxis_title="Date", yaxis_title="Percent Change from Preceding Period", xaxis_rangeselector_font_color='black',
                    xaxis_rangeselector_activecolor='#16FF32',
                    xaxis_rangeselector_bgcolor='white')
    P_C_fig.update_layout(template="none")
    P_C_fig.update_layout(hovermode="x unified")
    P_C_fig.update_yaxes(tick0=0, dtick=2000000)
    P_C_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(

            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),

                ]
            )
        ),
    )
    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Percent Change from Preceding Period,Seasonally Adjusted</h2>", 
    unsafe_allow_html=True)
    st.plotly_chart(P_C_fig, use_container_width=True, sharing="streamlit")

    df_P_C['Date'] = pd.to_datetime(df_P_C['Date']).dt.date
    df_P_C = df_P_C.iloc[::-1]
    df_P_C = df_P_C.reset_index(drop=True)   

    def convert_df_seasonal(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df_seasonal(df_P_C)
    st.download_button(
        label="Download data as CSV (Seasonally Adjusted)",
        data=csv,
        file_name='PC_Seasonally_Adjusted.csv',
        mime='text/csv',
)
    placeholder = st.empty()
    if st.checkbox("Show Dataframe (Seasonally Adjusted)"):
        st.dataframe(df_P_C.style.format({"% Change": '{:.1f}'}),use_container_width=True)  

        
    from api_calls import P_C_unseasonal_api
    df_P_C_NS=P_C_unseasonal_api()
    P_C_NS_fig = go.Figure()
    P_C_NS_fig = go.Figure()

    P_C_NS_fig.add_trace(
        go.Scatter(
            x=df_P_C_NS.Date,
            y=df_P_C_NS["% Change, Not Seasonal"],
            line_color="#16FF32",
        )
    )

    P_C_NS_fig.update_layout(xaxis_title="Date", yaxis_title="Percent Change from Preceding Period", xaxis_rangeselector_font_color='black',
                    xaxis_rangeselector_activecolor='#16FF32',
                    xaxis_rangeselector_bgcolor='white')
    P_C_NS_fig.update_layout(template="none")
    P_C_NS_fig.update_layout(hovermode="x unified")
    P_C_NS_fig.update_yaxes(tick0=0, dtick=2000000)
    P_C_NS_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(

            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),

                ]
            )
        ),
    )
    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Percent Change from Preceding Period, Not Seasonally Adjusted</h2>", 
    unsafe_allow_html=True)
    st.plotly_chart(P_C_NS_fig, use_container_width=True, sharing="streamlit")

    df_P_C_NS['Date'] = pd.to_datetime(df_P_C_NS['Date']).dt.date
    df_P_C_NS = df_P_C_NS.iloc[::-1]
    df_P_C_NS = df_P_C_NS.reset_index(drop=True)   

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_P_C_NS)
    st.download_button(
        label="Download data as CSV (Not Seasonally Adjusted)",
        data=csv,
        file_name='PC_Seasonally_Adjusted.csv',
        mime='text/csv',
)
    placeholder = st.empty()
    if st.checkbox("Show Dataframe (Not Seasonally Adjusted)"):
        st.dataframe(df_P_C_NS.style.format({"% Change, Not Seasonal": '{:.1f}'}),use_container_width=True)  
#End-------------------------------------------------------------------------------------


# Unemployment Rate-----------------------------------------------------------------------------------
def Unemp_states():
    options=["Select an index","Unemployment Rate (Seasonally Adjusted)","Unemployment Rate (Not Seasonally Adjusted)"]
    selection=st.selectbox('',options=options)
    return selection

# Unemployment Rate (Seasonally Adjusted)
def unempr_s():
    from api_calls import unempr_s_api
    df_combined_s=unempr_s_api()
    combined_s_fig = go.Figure()
    combined_s_fig.add_trace(
        go.Scatter(
            y=df_combined_s['texas_value'],
            x=df_combined_s.dates,
            name="Texas",
    #         line_color="#BAB0AC",
        )
    )

    combined_s_fig.add_trace(
        go.Scatter(
            x=df_combined_s.dates,
            y=df_combined_s['ohio_value'],
            name="Ohio",
    #         line_color="#0D2A63",
        )
    )

    combined_s_fig.add_trace(
        go.Scatter(
            x=df_combined_s.dates,
            y=df_combined_s["cali_value"],
            name="California",
    #         line_color="#16FF32",
        )
    )
    combined_s_fig.add_trace(
        go.Scatter(
            x=df_combined_s.dates,
            y=df_combined_s['ny_value'],
            name="New York",
    #         line_color="#0D2A63",
        )
    )
    combined_s_fig.add_trace(
        go.Scatter(
            x=df_combined_s.dates,
            y=df_combined_s['nj_value'],
            name="New Jersey",
    #         line_color="#0D2A63",
        )
    )
    combined_s_fig.add_trace(
        go.Scatter(
            x=df_combined_s.dates,
            y=df_combined_s['fl_value'],
            name="Florida",
    #         line_color="#0D2A63",
        )
    )
    combined_s_fig.add_trace(
        go.Scatter(
            x=df_combined_s.dates,
            y=df_combined_s['pn_value'],
            name="Pennsylvania",
    #         line_color="#0D2A63",
        )
    )

    # Formatting
    # fig.update_layout(width=,height=600)
    combined_s_fig.update_layout(xaxis_title="Date", yaxis_title="Unemployment_Rate")
    combined_s_fig.update_layout(template="plotly_dark")

    combined_s_fig.update_yaxes(tick0=2.0, dtick=0.8)
    combined_s_fig.update_layout(hovermode="x unified")
    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Unemployment Rate, Seasonally Adjusted</h2>", 
    unsafe_allow_html=True)
    st.plotly_chart(combined_s_fig, use_container_width=True, sharing="streamlit")

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_combined_s)
    st.download_button(
        label="Download data as CSV",
        data=csv,
       file_name='US_NotSeasonally_Adjusted.csv',
        mime='text/csv',
)
    placeholder = st.empty()
    if st.checkbox("Show Dataframe"):
        st.dataframe(df_combined_s,use_container_width=True)

#End-------------------------------------------------------------------------------------

# Unemployment Rate (Unseasonally Adjusted)
def unempr_us():
    from api_calls import unempr_us_api
    df_combined_us=unempr_us_api()
    combined_us_fig = go.Figure()
    combined_us_fig.add_trace(
        go.Scatter(
            y=df_combined_us['texas_value'],
            x=df_combined_us.dates,
            name="Texas",
    #         line_color="#BAB0AC",
        )
    )

    combined_us_fig.add_trace(
        go.Scatter(
            x=df_combined_us.dates,
            y=df_combined_us['ohio_value'],
            name="Ohio",
    #         line_color="#0D2A63",
        )
    )

    combined_us_fig.add_trace(
        go.Scatter(
            x=df_combined_us.dates,
            y=df_combined_us["cali_value"],
            name="California",
    #         line_color="#16FF32",
        )
    )
    combined_us_fig.add_trace(
        go.Scatter(
            x=df_combined_us.dates,
            y=df_combined_us['ny_value'],
            name="New York",
    #         line_color="#0D2A63",
        )
    )
    combined_us_fig.add_trace(
        go.Scatter(
            x=df_combined_us.dates,
            y=df_combined_us['nj_value'],
            name="New Jersey",
    #         line_color="#0D2A63",
        )
    )
    combined_us_fig.add_trace(
        go.Scatter(
            x=df_combined_us.dates,
            y=df_combined_us['fl_value'],
            name="Florida",
    #         line_color="#0D2A63",
        )
    )
    combined_us_fig.add_trace(
        go.Scatter(
            x=df_combined_us.dates,
            y=df_combined_us['pn_value'],
            name="Pennsylvania",
    #         line_color="#0D2A63",
        )
    )

    # Formatting
    # fig.update_layout(width=,height=600)
    combined_us_fig.update_layout(xaxis_title="Date", yaxis_title="Unemployment_Rate")
    combined_us_fig.update_layout(template="plotly_dark")

    combined_us_fig.update_yaxes(tick0=2.0, dtick=0.8)
    combined_us_fig.update_layout(hovermode="x unified")
    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Unemployment Rate, Not Seasonally Adjusted</h2>", 
    unsafe_allow_html=True)
    st.plotly_chart(combined_us_fig, use_container_width=True, sharing="streamlit")

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_combined_us)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='UR_Seasonally_Adjusted.csv',
        mime='text/csv',
)
    placeholder = st.empty()
    if st.checkbox("Show Dataframe"):
        st.dataframe(df_combined_us,use_container_width=True)

#End-------------------------------------------------------------------------------------


# --------------------------Real GDP-----------------------------------------------------
def gdp_plotting():
    from api_calls import Real_GDP_api
    gdp_df=Real_GDP_api()
    gdp_fig = go.Figure()
    gdp_fig.add_trace(
    go.Scatter(
        x=gdp_df.Date,
        y=gdp_df["Annualized QoQ Real GDP Growth"],
        name="Annualized QoQ Real GDP Growth",
        line_color="#16FF32"))

    gdp_fig.add_trace(
        go.Scatter(
            x=gdp_df.Date,
            y=gdp_df["YoY Real GDP Growth"],
            name="YoY Real GDP Growth",
            line_color="lightpink"))

    gdp_fig.update_layout(width=1400,height=600,xaxis_rangeselector_font_color='black',
                        xaxis_rangeselector_activecolor='#16FF32',
                        xaxis_rangeselector_bgcolor='white')
    gdp_fig.update_layout(template="plotly_dark")
    gdp_fig.update_yaxes(tickformat=".0%",range=[-0.15,0.15])
    gdp_fig.update_layout(margin_pad=15)
    gdp_fig.update_layout(hovermode="x unified")
    gdp_fig.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgb(60, 60, 58,0.1)')
    gdp_fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgb(60, 60, 58,0.1)')

    gdp_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(

            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),

                ]
            )
        ),
    )


    def threshhold(fig, variable, level, mode, fillcolor, layer):
        if mode == 'above':
            m = gdp_df[variable].gt(level)
        
        if mode == 'below':
            m = gdp_df[variable].lt(level)
            
        df1 = gdp_df[m].groupby((~m).cumsum())['Date'].agg(['first','last'])

        for index, row in df1.iterrows():
            fig.add_shape(type="rect",
                            xref="x",
                            yref="paper",
                            x0=row['first'],
                            y0=0,
                            x1=row['last'],
                            y1=1,
                            line=dict(color="rgb(60, 60, 58,0.1)",width=4,),
                            fillcolor=fillcolor,
                            layer=layer) 
        return(fig)
    
    gdp_fig = threshhold(fig = gdp_fig, variable = 'Annualized QoQ Real GDP Growth', level = 0, mode = 'below',
               fillcolor = 'rgb(60, 60, 58,0.1)', layer = 'below')

    gdp_fig = threshhold(fig = gdp_fig, variable = 'YoY Real GDP Growth', level = 0, mode = 'below',
               fillcolor = 'rgb(60, 60, 58,0.1)', layer = 'below')



    st.markdown("<h2 style='text-align: left; color: #16FF32;'>Real GDP Growth</h2>", 
    unsafe_allow_html=True)
    st.plotly_chart(gdp_fig, use_container_width=True, sharing="streamlit")

    gdp_df['Date'] = gdp_df['Date'].dt.strftime('%m-%d-%Y')

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(gdp_df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='GDP.csv',
        mime='text/csv')
    placeholder = st.empty()
    if st.checkbox("Show Dataframe"):
        st.dataframe(gdp_df.style.format({'GDP': '{:.0f}', 'Annualized QoQ Real GDP Growth': '{:.2f}', 'YoY Real GDP Growth': '{:.2f}'}),use_container_width=True)



# --------------------------END---------------------------------------------------------




# ------------------------------Payroll Benchmarks--------------------------------------
def industry_comparison():
    data=pd.read_csv('state_profiles_2017/data.csv')
    data=data.drop(['State','Employment\nRange Flag', 'Employment Noise Flag', 'Annual Payroll Noise Flag', 'Receipts\nNoise Flag'],axis=1)
    data['NAICS'] = data['NAICS'].str.replace(r'(\-\d+$)', '', regex=True)
    data['NAICS']=data['NAICS'].replace('--',0)
    data['NAICS']=data['NAICS'].astype(int)
    data=data[(data.NAICS.apply(lambda x: len(str(x))==1) | (data.NAICS.apply(lambda x: len(str(x))==2)))]
    data['Enterprise Size']=data['Enterprise Size'].str.replace(r'(^\d+\:)','',regex=True)
    data['Enterprise Size']= data['Enterprise Size'].str.lstrip()
    data.iloc[::,4:]=data.iloc[::,4:].replace(',','', regex=True)
    data.iloc[::,4:]=data.iloc[::,4:].astype('int64')
    data['Payroll/Employee']=(data['Annual Payroll\n($1,000)']/data['Employment'])*1000
    data['Payroll/Employee']=data['Payroll/Employee'].round(0)
    data['Payroll % of revenue']=((data['Annual Payroll\n($1,000)']*100)/data['Receipts\n($1,000)'])
    data['Payroll % of revenue']=data['Payroll % of revenue'].round(1)
    data[['Receipts\n($1,000)','Annual Payroll\n($1,000)','Payroll/Employee']]=data[['Receipts\n($1,000)','Annual Payroll\n($1,000)','Payroll/Employee']].replace(0, np.nan)
    data=data.dropna()
    data_enterpriseSize_Selection=data[data['Enterprise Size'].str.contains('Total')==False]
    enterprise_size_options=data_enterpriseSize_Selection['Enterprise Size'].unique()
    enterprise_size=st.selectbox('Select an enterprise size',options=enterprise_size_options,key='1')
    if enterprise_size:
        data=data[data['Enterprise Size']=='{}'.format(enterprise_size)]
        data_stateSelection=data[data['State Name'].str.contains('United States')==False]
        state_options=data_stateSelection['State Name'].unique()
        state_selected=st.selectbox('Select a State',options=state_options,key='15')
        state_default='United States'
        data_default=data[data['State Name'].str.contains("{}".format(state_default))==True]
        data=data[data['State Name'].str.contains("{}".format(state_selected))==True]
        if state_selected: 
            data_industrySelection=data[data['NAICS Description'].str.contains('Total')==False]
            industry_options_1=data_industrySelection['NAICS Description'].unique()
            industry_selected_1=st.selectbox('Select an Industry',options=industry_options_1,key='10',index=0)
            data_2=data_industrySelection[data_industrySelection['NAICS Description'].str.contains(industry_selected_1)==False]
            industry_options_2=data_2['NAICS Description'].unique()
            industry_selected_2=st.selectbox('Select an Industry',options=industry_options_2,key='11')
            data_3=data_2[data_2['NAICS Description'].str.contains(industry_selected_2)==False]
            industry_options_3=data_3['NAICS Description'].unique()
            industry_selected_3=st.selectbox('Select an Industry',options=industry_options_3,key='12')
            industry_default='Total'
            data=data[data['NAICS Description'].str.contains("{}|{}|{}|{}".format(industry_selected_1, industry_selected_2, industry_selected_3,industry_default))==True]
            data_default=data_default[data_default['NAICS Description'].str.contains("{}|{}|{}|{}".format(industry_selected_1, industry_selected_2, industry_selected_3,industry_default))==True]
            data= data.reindex(np.roll(data.index, shift=3))
            data_default= data_default.reindex(np.roll(data_default.index, shift=3))
            data=pd.concat([data_default,data],axis=0)
            data=data.drop('NAICS',axis=1)
            data[['Annual Payroll\n($1,000)','Receipts\n($1,000)','Payroll/Employee','Payroll % of revenue']]=data[['Annual Payroll\n($1,000)','Receipts\n($1,000)','Payroll/Employee','Payroll % of revenue']].astype(int)
            # data['Payroll % of revenue']=data['Payroll % of revenue'].astype(float)
            data['NAICS Description']= data['NAICS Description'].str.replace('Total', 'All Industries')
            data=data.reset_index(drop=True)
            return data,enterprise_size,industry_selected_1, industry_selected_2, industry_selected_3



def plotting_industryComparison(data_industryComparison):
    fig_industryComparison_payroll_employee = px.bar(data_industryComparison, x='NAICS Description', y="Payroll/Employee",
             color='State Name', barmode='group',text_auto=True,
             color_discrete_sequence=px.colors.qualitative.Set1)
    fig_industryComparison_payroll_employee.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,title='Payroll/Employee ({})'.format(enterprise_size))
    fig_industryComparison_payroll_employee.update_yaxes(title_font_family="Arial",tickformat=",.0f",tickprefix="$")
    fig_industryComparison_payroll_employee.update_xaxes(title='Industry')
    st.plotly_chart(fig_industryComparison_payroll_employee, use_container_width=True, sharing="streamlit")


    fig_industryComparison_payroll_percent_revenue = px.bar(data_industryComparison, x='NAICS Description', y='Payroll % of revenue',
             color='State Name', barmode='group',text_auto=True,
             color_discrete_sequence=px.colors.qualitative.Dark2)
    fig_industryComparison_payroll_percent_revenue.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,title='Payroll % of revenue ({})'.format(enterprise_size))
    fig_industryComparison_payroll_percent_revenue.update_yaxes(title_font_family="Arial",tickformat=",.0f",ticksuffix="%")
    fig_industryComparison_payroll_percent_revenue.update_xaxes(title='Industry')
    st.plotly_chart(fig_industryComparison_payroll_percent_revenue, use_container_width=True, sharing="streamlit")

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(data_industryComparison)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Industry_payrollBenchmarks.csv',
        mime='text/csv')

    st.dataframe(data_industryComparison,use_container_width=True)
    st.markdown(""" <style> .font {
        font-size:20px} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Source: 2017 Statistics of U.S Businesses from US Census Bureau</p>', unsafe_allow_html=True)
    
# --------------The End---------------------------------------------------------------------------------------------------------





# -----------------------------------Payroll Banchmarks (subindustry)----------------
def payroll_benchmarks_subindustry():
    data=pd.read_csv('state_profiles_2017/data.csv')
    data=data.drop(['State','Employment\nRange Flag', 'Employment Noise Flag', 'Annual Payroll Noise Flag', 'Receipts\nNoise Flag'],axis=1)
    # data['NAICS'] = data['NAICS'].str.replace(r'(\-\d+$)', '', regex=True)
    data['NAICS']=data['NAICS'].replace('--',0)
    data=data[(data.NAICS.apply(lambda x: len(str(x))==2) | (data.NAICS.apply(lambda x: len(str(x))==4) | (data.NAICS.apply(lambda x: len(str(x))==5))))]
    data['Enterprise Size']=data['Enterprise Size'].str.replace(r'(^\d+\:)','',regex=True)
    data['Enterprise Size']= data['Enterprise Size'].str.lstrip()
    data.iloc[::,4:]=data.iloc[::,4:].replace(',','', regex=True)
    data.iloc[::,4:]=data.iloc[::,4:].astype('int64')
    data['Payroll/Employee']=(data['Annual Payroll\n($1,000)']/data['Employment'])*1000
    data['Payroll/Employee']=data['Payroll/Employee'].round(0)
    data['Payroll % of revenue']=((data['Annual Payroll\n($1,000)']*100)/data['Receipts\n($1,000)'])
    data['Payroll % of revenue']=data['Payroll % of revenue'].round(1)
    data[['Receipts\n($1,000)','Annual Payroll\n($1,000)','Payroll/Employee']]=data[['Receipts\n($1,000)','Annual Payroll\n($1,000)','Payroll/Employee']].replace(0, np.nan)
    data=data.dropna()
    data_enterpriseSize_Selection=data[data['Enterprise Size'].str.contains('Total')==False]
    enterprise_size_options=data_enterpriseSize_Selection['Enterprise Size'].unique()
    enterprise_size=st.selectbox('Select an enterprise size',options=enterprise_size_options,key='1')
    if enterprise_size:
        state_default='United States'
        data_USA=data[data['State Name'].str.contains('{}'.format(state_default))==True]
        data=data[data['State Name'].str.contains('{}'.format(state_default))==False]
        data=data[data['Enterprise Size']=='{}'.format(enterprise_size)]
        data_USA=data_USA[data_USA['Enterprise Size']=='{}'.format(enterprise_size)]
        # data_NAICS=data[data['NAICS Description'].str.contains('{}'.format('Total'))==False]       #Removing Total from NAICS Description so that in options Total will not display#
        data_new=data.copy()
        data_new['NAICS'] = data_new['NAICS'].str.replace(r'(\-\d+$)', '', regex=True)
        data_new=data_new[(data_new.NAICS.apply(lambda x: len(str(x))==2))]
        industry_options=data_new['NAICS Description'].unique()
        industry_selected=st.selectbox('Select an Industry',options=industry_options,key='2')

        if industry_selected:
            data_NAICS=data[data['NAICS Description']=='{}'.format(industry_selected)]
            subIndustry_NAICS=data_NAICS['NAICS'].iloc[0]
            subIndustry_NAICS=subIndustry_NAICS.replace('-','|')
            subindustry_NAICS_data=data[data.NAICS.str.contains('^({})[0-9][0-9]$'.format(subIndustry_NAICS), regex= True,na=False)]
            subIndustry_options=subindustry_NAICS_data['NAICS Description'].unique()
            subIndustry_selected=st.selectbox('Select an Sub-Industry',options=subIndustry_options,key='3')

            if subIndustry_selected:
                data=data[(data.NAICS.apply(lambda x: len(str(x))==4))]
                data=data[data['NAICS Description']=='{}'.format(subIndustry_selected)]
                data_USA=data_USA[data_USA['NAICS Description']=='{}'.format(subIndustry_selected)]
                state_1_options=data['State Name'].unique()
                state_1_selected=st.selectbox('Select an State',options=state_1_options,key='4')
                data_state_2=data[data['State Name'].str.contains('{}'.format(state_1_selected))==False]
                state_2_options=data_state_2['State Name'].unique()
                state_2_selected=st.selectbox('Select an State',options=state_2_options,key='5')
                data_state_3=data_state_2[data_state_2['State Name'].str.contains('{}'.format(state_2_selected))==False]
                state_3_options=data_state_3['State Name'].unique()
                state_3_selected=st.selectbox('Select an State',options=state_3_options,key='6')
                data=data[data['State Name'].str.contains("{}|{}|{}".format(state_1_selected, state_2_selected, state_3_selected))==True]
                data_USA=data_USA[(data_USA.NAICS.apply(lambda x: len(str(x))==4))]
                data=pd.concat([data_USA,data],axis=0)
                data=data.drop(['NAICS'],axis=1)
                data= data.reindex(np.roll(data.index, shift=3))
                data=data.reset_index(drop=True)
                return data,enterprise_size

def plotting_stateComparison(data_stateComparison):
    
    bar_colors = ["azure", 'cyan',"lime","deeppink"]
    fig_stateComparison_payroll_employee = px.bar(data_stateComparison, x='State Name', y="Payroll/Employee",text_auto=True,color='State Name',color_discrete_sequence=bar_colors,opacity=1)
    fig_stateComparison_payroll_employee.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,title='Payroll/Employee ({})'.format(enterprise_size))
    fig_stateComparison_payroll_employee.update_yaxes(title_font_family="Arial",tickformat=",.0f",tickprefix="$")
    fig_stateComparison_payroll_employee.update_xaxes(title='State')
    st.plotly_chart(fig_stateComparison_payroll_employee, use_container_width=True, sharing="streamlit")

    colors = ["#7CEA9C", '#50B2C0', "hsv(348, 66%, 90%)", "hsl(45, 93%, 58%)"]
    fig_stateComparison_payroll_percent_revenue = px.bar(data_stateComparison, x='State Name', y='Payroll % of revenue',color='State Name',text_auto=True,color_discrete_sequence=px.colors.qualitative.Light24,opacity=1)
    fig_stateComparison_payroll_percent_revenue.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,title='Payroll % of revenue ({})'.format(enterprise_size))
    fig_stateComparison_payroll_percent_revenue.update_yaxes(title_font_family="Arial",tickformat=",.0f",ticksuffix="%")
    fig_stateComparison_payroll_percent_revenue.update_xaxes(title='State')
    st.plotly_chart(fig_stateComparison_payroll_percent_revenue, use_container_width=True, sharing="streamlit")

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(data_stateComparison)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='State_payrollBenchmarks.csv',
        mime='text/csv')
    st.dataframe(data_stateComparison.style.format({'Annual Payroll\n($1,000)': '{:.0f}', 'Receipts\n($1,000)': '{:.0f}', 'Payroll/Employee': '{:.0f}','Payroll % of revenue':'{:.1f}'}),use_container_width=True)
    st.markdown(""" <style> .font {
    font-size:20px} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Source: 2017 Statistics of U.S Businesses from US Census Bureau</p>', unsafe_allow_html=True)

#  -------------------------End--------------------------------------------------











#  Main Logic
# Set the pages and load in respective data
if selected == '10 Year Minus 2 Year Treasury': 
    TenYrTwoYr()
if selected == 'Consumer Sentiment':
    consumer_sentiment()
if selected == 'Unemployment Claims': 
    unemployment()
if selected == 'Price Index':
    PI_selection=CPI()
    if PI_selection == 'Year over Year Percent Change CPI':
        YoYPC()
    if PI_selection == 'Producer Price Index by Commodity: All Commodities':
        ppi()
    if PI_selection == 'Employment Cost Index: Total compensation: All Civilian':
        eci()
if selected == 'Average House Sales Price':
    house_price()
if selected== "Retail Trade and Food Services":
    R_selection=Retail()
    if R_selection == 'Millions of Dollars':
        MD()
    if R_selection=='Percent Change from Preceding Period':
        P_C()
if selected =="Unemployment Rates for States":
    UnempR_selection=Unemp_states()
    if UnempR_selection == "Unemployment Rate (Seasonally Adjusted)":
        unempr_s()
    if UnempR_selection=="Unemployment Rate (Not Seasonally Adjusted)":
        unempr_us()

if selected=="Real GDP Growth":
    gdp_plotting()

if selected=="Payroll Benchmarks":
    data_industryComparison,enterprise_size,industry_selected_1, industry_selected_2, industry_selected_3=industry_comparison()
    if st.button(label="Check"):
        plotting_industryComparison(data_industryComparison)

if selected=="Payroll Benchmarks (Sub Industry)":
    data_stateComparison,enterprise_size=payroll_benchmarks_subindustry()
    if st.button(label="Check"):
        plotting_stateComparison(data_stateComparison)
    

    






















