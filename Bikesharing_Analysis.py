import time
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from matplotlib.ticker import ScalarFormatter
from models import preprocess
from models import linreg_model
from models import enet_model
from models import rf_model
from models import dt_model
from models import knn_model
from models import gb_model
from models import create_predictions_df
from sklearn.tree import plot_tree


st.set_page_config(page_title="Bike Sharing: Washington DC", layout="wide", page_icon=":bike:")
st.markdown("<h1 style='text-align: center;'>&#x1F6B2 Bike Sharing: Washington DC &#x1F6B2</h1>", unsafe_allow_html=True)

st.image("https://s3.ca-central-1.amazonaws.com/ehq-production-canada/redactor_assets/assets/4bddbb00662f81e2f2b671ec78db406f2ec85019/000/015/250/original/12_2019_Enviro_BikeShare_GetInvolved_Banner.jpg?1575667577", use_column_width=True)

# read in data
df=pd.read_csv("bike-sharing_hourly.csv")

# convert dteday to datetime format
df["dteday"] = pd.to_datetime(df["dteday"])
# create year column
df["year"] = df["dteday"].dt.year

df.rename(columns={
        'dteday' : 'Date',
        'season' : 'Season',
        'yr' : 'Year',
        'mnth' : 'Month',
        'hr' : 'Hour',
        'holiday' : 'Is_Holiday',
        'weekday' : 'Weekday',
        'workingday' : 'Is_Working_Day',
        'weathersit' : 'Weather_Condition',
        'temp' : 'Temperature',
        'atemp' : 'Air_Temperature',
        'hum' : 'Humidity',
        'windspeed' : 'Windspeed',
        'casual' : 'Casual_Users',
        'registered' : 'Registered_Users',
        'cnt' : 'Total_Users'
    }, inplace=True)

############### PART I ###############

def parti_page():
    #title
    st.markdown("<h2 style='text-align: center;'>Part I: Introduction</h2>", unsafe_allow_html=True)

    #Description
    st.write('''
    Hi! As a consultancy firm, we have been hired by the local government in Washington D.C to conduct a comprehensive analysis of the bike-sharing service in the city. The aim of this analysis is to provide insights into how citizens are using the bike-sharing service, which will help in optimizing costs and improving the service. We divide this aim into the two main goals below.
    ''')
    
    # goal subheader
    st.subheader("Goals")

    # write goal 1
    st.write("1. Provide a deep analysis of the bike-sharing service in Washington DC along with recommendations for optimizing costs and providing a better service. Our analysis will focus on understanding the patterns and trends in the usage of the bike-sharing service. We will also identify any gaps in the service and suggest improvements to optimize its provision.")
    
    # write goal 2
    st.write("2. Create a machine learning model to predict the total number of bicycle users on an hourly basis to help in optimization of bike provisioning optimization of costs incurred from the outsourcing transportation company.")
    
    st.markdown('''
    This app will provide an interactive analysis of the bike-sharing service in the city. It is divided into 4 parts:
    - Basic Exploratory Analysis: initial data exploration and insights
    - Customer Behavior Analysis: deeper data exploration with insights on each type of bike user
    - Model Building: process of creating our final machine learning model
    - Model Predictions: where you can input your own data ang generate a prediction for total bike users using our model
    ''')
    
############### PART II ###############
def partii_page():
    #title
    st.markdown("<h2 style='text-align: center;'>Part II: Basic Data Exploration</h2>", unsafe_allow_html=True)
    
            # write description of splitting variabe types
    st.write("Here, we first describe the dataset. We then split our columns into categorical and numerical and computed descriptive statistics. We also created box plots to analyze the distribution of the variables visually and bar charts to view each variables relationship with the target variable. All of these can be found in the categorical and numerical feature tabs.")
    
    tab1, tab2, tab3 = st.tabs(["Dataset Description", "Categorical Features", "Numeric Features"])
    
    with tab1:
        
        st.subheader("Dataset Description")
        
        # write description 
        st.text("The dataset used contains information about bike rentals in Washington DC. It contains data from the years 2011 and 2012 and contains the features below.")

        # table with dataset features
        markdown_text = """\
        | Column Name | Description |
        | --- | --- |
        | `instant` | Record index |
        | `Date` | Date |
        | `Season` | 1: spring<br>2: summer<br>3: fall<br>4: winter |
        | `Year` | 0: year 2011<br> 1: year 2012 |
        | `Month` | Month (1 to 12) |
        | `Hour` | Hour (0 to 23) |
        | `Is_Holiday` | Whether day is holiday or not |
        | `Weekday` | Day of the week |
        | `Is_Working_Day` | If day is neither weekend nor holiday is 1, otherwise is 0. |
        | `Weather_Condition` | 1: Clear, Few clouds, Partly cloudy, Partly cloudy<br>2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist<br>3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds<br>4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog |
        | `Temperature` | Normalized temperature in Celsius. The values are divided to 41 (max) |
        | `Air_Temperature` | Normalized feeling temperature in Celsius. The values are divided to 50 (max) |
        | `Humidity` | Normalized humidity. The values are divided to 100 (max) |
        | `Windspeed` | Normalized wind speed. The values are divided to 67 (max) |
        | `Casual_Users` | Count of casual users |
        | `Registered_Users` | Count of registered users |
        | `Total_Users` | Count of total rental bikes including both casual and registered |
        """
        st.write(markdown_text, unsafe_allow_html=True)

        # dataset preview
        st.text("Below is a preview of the dataset.")
        st.write(df.head())

        # check dataset shape
        st.text("The dataset contains 17379 rows, 17 columns and has no null values as seen below")

        st.code('''
        # Check shape
        df.shape
        ''')
        st.write(df.shape)

        # check dataset shape
        st.code('''
        #Check for null values
        df.isnull().sum()
        ''')
        st.write(df.isnull().sum().sum())
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        fig = px.imshow(df.corr(), color_continuous_scale='YlGnBu')
        fig.update_layout(
        width=1500,
        height=600,
        title='Correlation Heatmap')
        st.plotly_chart(fig, use_container_width=True)

        # write correlation findings
        st.write('''
        Key findings from correlation analysis:
        - Season is positively correlated with months
        - Workingday is correlated with holiday which makes sense as all holidays are not working days, so one of these variables should be removed to avoid introducing irrelevant noisy features into the model.
        - Many features are negatively correlated with humidity: hour, registered users, casual users, and windspeed so might be useful to remove humidity from the model. However, it has a strong negative correlation with the target variable, suggesting it could be useful to keep in.
        - Working day is negatively correlated with casual because casual suers use less bikes on work days 
        - Registered and casual have to be removed, as these are a subset of the target.
        ''')
    
    with tab2:
        
        st.subheader("Categorical Features")

        # Descriptive stats of categorical columns
        categorical_cols = df[["Weekday", "Season", "Year", "Is_Holiday","Weather_Condition","Is_Working_Day"]]

        st.code('''
        #Descriptive statistics of categorical features
        categorical_cols.describe()
        ''')
        st.write(categorical_cols.describe())

        #Plot categorical columns
        fig = px.box(categorical_cols, color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, range=[0, None]))
        st.plotly_chart(fig, use_container_width=True)

        # plot categorical cols againt target variable
        selected_category = st.selectbox("Select a categorical feature to view its relationship with the target variable, total users", categorical_cols.columns)
        fig = go.Figure(go.Bar(x=categorical_cols[selected_category], y=df['Total_Users']))

        fig.update_layout(
            xaxis_title=selected_category,
            yaxis_title='Bike rentals',
            width=400,
            height=200,
            margin=dict(l=50, r=50, t=30, b=20),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, range=[0, None]))
        st.plotly_chart(fig, use_container_width=True)

        # categorical variables insights
        st.write('''
        Categorical variables key findings:
        - There seems to be a similar distribution across weekdays
        - There are 4 seasons as expected with similar distribution across them
        - We only have data for 2 years, encoded as 0 and 1. There are more rentals in 2012 than in 2011
        - There are many more rentals on non-holidays than holidays
        - There are few data points for harsh weather conditions, so less people rent bikes when there are harsh weather conditions
        ''')

    with tab3:
        
        st.subheader("Numerical Features")
        # Descriptive stats of numerical columns
        numerical_cols = df[["Temperature", "Air_Temperature", "Humidity", "Windspeed", "Casual_Users", "Registered_Users", "Month", "Hour",]]

        st.code('''
        #Descriptive statistics of numerical features
        numerical_cols.describe()
        ''')
        st.write(numerical_cols.describe())

        #Plot numerical columns
        selected_num_category = st.selectbox("Select numerical features", ("casual & registered users", "Month & hour", "weather factors"))

        if selected_num_category == "casual & registered users":
            numerical_cols_boxplot = df[["Casual_Users", "Registered_Users"]]
        elif selected_num_category == "Month & hour":
            numerical_cols_boxplot = df[["Month", "Hour"]]
        elif selected_num_category == "weather factors":
            numerical_cols_boxplot = df[["Temperature", "Air_Temperature", "Humidity", "Windspeed"]]
        else:
            numerical_cols_boxplot = numerical_cols

        fig = px.box(numerical_cols_boxplot, color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, range=[0, None]))
        st.plotly_chart(fig, use_container_width=True)

        # plot numerical cols againt target variable
        selected_numerical = st.selectbox("Select a numerical feature to view its relationship with the target variable, total users", numerical_cols.drop(columns=["Casual_Users", "Registered_Users"]).columns)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=numerical_cols[selected_numerical], y=df['Total_Users'], mode='markers', name=selected_numerical))

        fig.update_layout(
            xaxis_title=selected_numerical,
            yaxis_title="Bike rentals",
            height=400, width=600
        )
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, range=[0, None]))
        st.plotly_chart(fig, use_container_width=True)

        # numerical variables insights
        st.write('''
        Numerical variables key findings:
        - There are more registered than casual users but both have quite a few outliers who rent bikes very often
        - Month and hour do not have any significant outliers
        - Temperature and air temperature have similar distributions and no significant outliers
        - Humidity has one outliers, with very low humidity
        - Windspeed has quite a few outliers with high windspeed
        ''')

############### PART III ###############
def partiii_page():
    #title
    st.markdown("<h2 style='text-align: center;'>Part III: Customer Behavior Analysis 2011-2012</h2>", unsafe_allow_html=True)

    # write description 
    st.write("Below you can find an interactive analysis of registered vs casual bike users along with relevant insights. To see recommendations on how to optimize the bike service based on these insights, navigate to the 'Recommendations' tab.")
     
    tab1, tab2, tab3, tab4 = st.tabs(["Users over Time", "Users by Categorical Features", "Users by Numeric Features", "Recommendations"])

    with tab1: 
        
        st.write("Select a date range and time dimension to analyze the distribution of bike users over time.")

        # add calendar for start and end dates
        date_col_1, date_col_2 = st.columns(2)
        with date_col_1:
            start_date = st.date_input("Start date", value=df["Date"].min(), min_value=df["Date"].min(),max_value=df["Date"].max())
        with date_col_2:
            end_date = st.date_input("End date", value=df["Date"].max(), min_value=df["Date"].min(), max_value=df["Date"].max())

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # filter data based on selected dates
        partiii_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        partiii_df["hour"] = partiii_df["Date"] + pd.to_timedelta(partiii_df["Hour"], unit="h")
        partiii_df['month'] = partiii_df['Date'].dt.month
        partiii_df['day'] = partiii_df['Date']

        # score cards
        registered_users = int(partiii_df["Registered_Users"].sum())
        casual_users = int(partiii_df["Casual_Users"].sum())
        total_users = int(partiii_df["Total_Users"].sum())

        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            st.metric(label="Registered Users", value="{:,}".format(registered_users))
        with middle_column:
            st.metric(label="Casual Users", value="{:,}".format(casual_users))
        with right_column:
            st.metric(label="Total Users", value="{:,}".format(total_users))

        # title for time series graph
        st.subheader("Bike Rentals over Time")

        # select granularity of time series plot
        selected_timeframe = st.radio("", ["hour", "day", "month"], horizontal=True)

        #group df by chosen granularity
        df_grouped = partiii_df.groupby(selected_timeframe).sum().reset_index()

        # create figure with total number of users
        fig = px.line()

        # add trace for total users
        total_trace = go.Scatter(x=df_grouped[selected_timeframe], y=df_grouped["Total_Users"], name='Total Users')
        fig.add_trace(total_trace)

        # add trace for casual users
        casual_trace = go.Scatter(x=df_grouped[selected_timeframe], y=df_grouped["Casual_Users"], name='Casual Users')
        fig.add_trace(casual_trace)

        # add trace for regular users
        registered_trace = go.Scatter(x=df_grouped[selected_timeframe], y=df_grouped["Registered_Users"], name='Registered Users')
        fig.add_trace(registered_trace)

        #configure figure layout
        fig.update_yaxes(title="Total Bikes Rented")
        fig.update_xaxes(title="Day")
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, range=[0, None]))
        fig.update_layout(legend=dict(title='Type of User'))

        # show the plot
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        
        st.subheader("Categorical Features Analysis")
        
        st.write("Select a categorical feature to analyze the distribution of bike users against the selected feature.")

        # define function to plot bar plot of categorical variables by number of registered and casual users
        def plot_grouped_data(partiii_df, groupby_col, data_col1, data_col2):
            grouped = partiii_df.groupby(groupby_col).sum()

            fig = go.Figure()

            n_categories = len(grouped.index)
            width = 0.5 / n_categories  # width of the bars, adjusted for number of categories
            offset = width / 2  # offset for positioning the bars

            for i, col in enumerate([data_col1, data_col2]):
                positions = [j + (i - 0.5) * width for j in range(n_categories)]
                fig.add_trace(go.Bar(x=grouped.index, y=grouped[col], name=col.capitalize(),
                                     xaxis='x', yaxis='y'))

            fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(n_categories)),
                                         ticktext=grouped.index, showgrid=False),
                              yaxis=dict(tickformat='.2f', showgrid=False),
                              barmode='group', width=800, height=300, legend=dict(x=0.01, y=0.99),
                              plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0),
                              )
            return fig

        # create sleection box for categorical features
        selected_cat_feat = st.selectbox("Categorical Feature Selection", ("Is_Working_Day", "Season", "Weekday", "Year", "Is_Holiday", "Weather_Condition"))

        # create a for loop to display the description for each plot
        cat_plot_description = {
            "Is_Working_Day": "The amount of casual users is similar in working days and non working days, while the total amount of registered users clearly increases in working days",
            "Season": "Fall shows the most amount of both registered and casual, while Summer and Winter have a similar amount of casual users, the decrease in these users is clear for the Winter Season",
            "Weekday": "Registered users have an increase during weekdays, opposite to casual users which increase during the weekends",
            "Year": "There was an increase in overall users for 2011, most of this growth was driven by an increase in registered users",
            "Is_Holiday": "The total amount of users is almost equal to zero during the holidays, bike sharing use doesn't seem to be common during festivities",
            "Weather_Condition": "The harshed the weather conditions become, the more the total amount of users decreases. Heavy rain seems to deter bike sharing use",
        }

        # create chart title and decription based on chosen variable
        selected_cat_feat_recoded = selected_cat_feat.replace("_", " ")
        st.subheader(f"User Type by {selected_cat_feat_recoded}")

        # call the function and display the plot
        fig_cat = plot_grouped_data(partiii_df, selected_cat_feat, "Casual_Users", "Registered_Users")
        st.plotly_chart(fig_cat, use_container_width=True)

        # create decription based on chosen variable
        st.caption(cat_plot_description[selected_cat_feat])


    with tab3:
    
        st.subheader("Numerical Features Analysis")
        
        st.write("Select a numerical feature to analyze the distribution of bike users against the selected feature.")
        
        # define function to plot numerical variables by number of registered and casual users
        def plot_grouped_data(partiii_df, groupby_col, data_col1, data_col2):
            grouped = partiii_df.groupby(groupby_col).sum().reset_index()

            fig = px.scatter(grouped, x=groupby_col, y=[data_col1, data_col2],
                             labels={groupby_col: groupby_col.capitalize(), 
                                     'value': 'Total'})

            fig.update_layout(xaxis_tickangle=-45)
            fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            return fig

        # create sleection box for numerical features
        selected_num_feat = st.selectbox("Numerical Features", ("Temperature", "Air_Temperature", "Humidity", "Windspeed", "Month", "Hour"))

        # create a for loop to display the description for each plot
        num_plot_description = {
            "Temperature": "Casual and registered users per temperature follow a similar distribution, except that there are many more registered than casual users. There are few users at low and high temperatures, and most users are concentrated around mid to high temperatures.",
            "Air_Temperature": "Most bike rentals occur in the mid range air temperatures. Additionally, there is a spike of casual usrs when the air temperature is around 0.65.",
            "Humidity": "There are few bike rentals when the humidity is very low. The number of bike rentals increases as humidity increases until a certain point and then decreases again.",
            "Windspeed": "There are many more bike rentals at lower windspeeds than there are at higher windspeeds, both for casual and registered users.",
            "Month": "There are more bike rentals in the summer months, presumably when it is warmer and the weather is more suitable for biking.",
            "Hour": "The distribution of casual and registered users per hour is different. Casual users rent bikes throughout the day, increasing from the morning until about 3:00pm and then decreasing again, showing a smooth pattern. Registered users on the other hand rent many bikes around 8:00am and again around 5:00pm, suggesting that many of these users rent bikes to go to and from work.",
        }

        # create chart title and decription based on chosen variable
        selected_num_feat_recoded = selected_num_feat.replace("_", " ")
        st.subheader(f"User Type by {selected_num_feat_recoded}")

        # call the function and display the plot
        fig = plot_grouped_data(partiii_df, selected_num_feat, "Casual_Users", "Registered_Users")
        st.plotly_chart(fig, use_container_width=True)

        # create decription based on chosen variable
        st.caption(num_plot_description[selected_num_feat])

    with tab4:
        
        # Recommendations
        st.subheader("Recommendations for Optimizing Bike Rental Service")
        st.write("Based on insights from the plots above, we suggest the recommendations below. By implementing these recommendations, bike sharing services could better match supply with demand, improve user experience, and save costs.")
        st.markdown("<h5 style='text-align: left;'>1. Optimize bike availability</h5>", unsafe_allow_html=True)
        st.markdown('''
        Bike sharing services are designed to provide users with a convenient and flexible mode of transportation. If there aren't enough bikes available for users when they need them, it can lead to frustration and dissatisfaction with the service. Some ways to optimize bike availability are:
        - If data on the location of each bike station were provided, the service would be able to regularly redistribute bikes from low-demand areas to high-demand areas to ensure that there are always bikes available where users need them.
        - Increase the number of bikes available during peak usage times, such as rush hour (mornings around 8am and afternoons 5pm), to meet demand.
        ''')
        st.markdown("<h5 style='text-align: left;'>2. Use dynamic pricing/incentives</h5>", unsafe_allow_html=True)
        st.markdown('''
        Introduce dynamic pricing that can be adjusted based on demand to incentivize users to rent bikes during off-peak hours and balance out the demand for bikes. For example:
        - Charge a premium during rush hour to even out demand. 
        - Charge less per minute for the first 15 minutes to incentivize shorter distances and ensure bike availability to more users.
        - Charge less during winter months/rainy or windy days to encourage users to rent bikes.
        - Offer a price incentive during non-working days for users to drop bikes in stations that need a minimum number of bikes, that way, reducing the need for manual redistribution by service staff.
    Leverage the fact that so many users are registered in your system to keep them updated on the latest the latest promotions and deals.
        ''')
        st.markdown("<h5 style='text-align: left;'>3. Optimize Costs during low Demand Periods</h5>", unsafe_allow_html=True)
        st.markdown('''
        During the colder months in Washington or festivities, when the number of users decreases significantly, consider reducing the number of bikes available to match demand and save costs.
        - By retrieving a certain of number of bikes during low demand peaks you also extend the usage life of the bike. These happen to be the periods when weather conditions are harsher, making it particularly important to protect the bikes.
        - Take advantage of low-demand periods to carry out maintenance on bikes and stations, which will help to maintain the service at optimum levels and prevent losses due to equipment breakdown.
        ''')

############### PART III ###############
def partiv_page():
    # title
    st.markdown("<h2 style='text-align: center;'>Part IV: Model Building</h2>", unsafe_allow_html=True)

    # write description 
    st.write('''
    To predict the total number of bicycle users on an hourly basis, we built a regression model with "cnt" (total number of users as the target variable. To build we conductedthe steps below.
    ''')
    
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Engineering", "Data Processing", "Models", "Final Model (further information)"])
    
    with tab1:
    
        # feature engineering
        st.subheader("Feature Engineering")
    
        st.write("Based on business knowledge we created the follong new features:")

        # display new features in table
        markdown_text = """\
        | Column Name | Description |
        | --- | --- |
        | `month_start` | 0: not first day of month<br>1: first day of month |
        | `month_end` | 0: not last day of month<br>1: last day of month |
        | `quarter` | 1: first quarter<br>2: second quarter<br>3: third quarer<br>4: fourth quarter |
        | `is_weekend` | 0: weekdays<br> 1: weekend days |
        | `day_period` | 1: 00:00 - 5:00 <br>2: 6:00 - 11:00 <br>3. 12:00 - 17:00<br>4: 18:00 - 23:00 |
        | `weather_factor` | weathersit x windspeed |
        | `min_daily_temp`<br>`ytd_min_daily_temp` | Minimum temperature of the current and previous day |
        | `max_daily_temp`<br>`ytd_max_daily_temp` | Maximum temperature of the current and previous day |
        | `mean_daily_temp`<br>`ytd_mean_daily_temp` | Average temperature of the current and previous day |
        | `min_daily_hum`<br>`ytd_min_daily_hum` | Minimum humidity of the current and previous day |
        | `max_daily_hum`<br>`ytd_max_daily_hum` | Maximum humidity of the current and previous day |
        | `mean_daily_hum`<br>`ytd_mean_daily_hum` | Average humidity of the current and previous day |
        | `cos_hour` | cosine of `hr`|
        | `cos_month` | cosine of `mnth` |
        | `cos_day` | cosine of `weekday` |
        | `is_rush_hour` | 0: Non-rush hours<br>1: Rush hours (8:00,17:00,18:00) |
        | `is_night` | 0: During the day<br>1: During the early hours of the morning (3:00-5:00) |
        """
        st.write(markdown_text, unsafe_allow_html=True)
    
    with tab2:
    
        # train test split
        st.subheader("Train Test Split")
        st.write("We sorted the dataset in ascending order by date and used a time series split to create a train (first 80% of data), a validation set (next 10% of data) and a test set (final 10% of data). You can find the code to do this below.")

        # display code used for train test split
        st.code('''
        from sklearn.model_selection import train_test_split

        # establish proportions for each set
        training_proportion = 0.80  # 80% for training
        validation_proportion = 0.10  # 10% for validation
        test_proportion = 0.10  # 10% for testing

        # create training and test sets
        X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(x_df, y_df, test_size=test_proportion, shuffle=False)

        # create training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_and_val, y_train_and_val, test_size=validation_proportion/(training_proportion+validation_proportion), shuffle=False)
        ''')

        # feature processing
        st.subheader("Feature Processing")
        st.write("We one hot encoded categorical variables and scaled numerical variables using column transformers and pipelines, as you can see below. We used these transformations for linear regression, Elastic Net, and KNN. The other models we tried are robust to these transformations so we did not perform them on the data.")

        # display code used for processing
        st.code('''
            # Create pipeline for each data type
        numerical_transformer = Pipeline(steps=[('num_scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder())])

        # Define the preprocessor, specifying which pipeline to apply to each data type
        preprocessor = ColumnTransformer(transformers=[
            ('numerical_transformer', numerical_transformer, num_feat),
            ('categorical_transformer', categorical_transformer, cat_feat)])
        ''')

        # return datasets from pre-processing
        X_train, y_train, X_test, y_test, X_train_processed, X_test_processed, test_dates = preprocess()
    
    
    with tab3:
    
        # models attempted
        st.subheader("Models Attempted")

        # wtrite description of hyper parameter tuning
        st.markdown('''
        We attempted the following models:
        - Linear Regression
        - K Nearest Neighbors
        - Decision Tree
        - Random Forest
        - Elastic Net
        - Gradient Boosting (final model)
        ''')
        st.write("To tune the hyperparameters for each of these models, we ran a grid search with k fold cross validation with 5 folds. We then trained the model with the best parameters found in the grid search on the entire training set and used it to predict the total number of bike rentals on the validation set. We then chose the model with the best performance on the train and validation set to use as our final model (gradient boosting). You can see the model performance on train and test sets for each of the models attempted below.")

        # model descriptions
        chosen_model = st.selectbox("Choose a model", ["Linear Regression", "K Nearest Neighbors", "Decision Tree", "Random Forest", "Elastic Net", "Gradient Boosting"])

        model_performance_description = ""

        if chosen_model == "Linear Regression":
            metrics, y_test_predictions = linreg_model(X_train_processed, y_train, X_test_processed, y_test)
            model_performance_description = "Linear regression is a simple and interpretable model, but it may not capture complex nonlinear relationships in our data given the low scores obtained."
        elif chosen_model == "Elastic Net":
            metrics, y_test_predictions = enet_model(X_train_processed, y_train, X_test_processed, y_test)
            model_performance_description = "Elastic net was chosen to compare it's results to Random Forest's, given that both help reducing overfitting, it was important to compare how a regularized linear regression model performed vs a Desicion Tree model."
        elif chosen_model == "Decision Tree":
            metrics, y_test_predictions = dt_model(X_train, y_train, X_test, y_test)
            model_performance_description = "Decision Tree was found to be overfitting on the training set before reducing the model's complexity. Next step was to find models that helped us reduce overfitting while maximizing the Test score."
        elif chosen_model == "K Nearest Neighbors":
            metrics, y_test_predictions = knn_model(X_train_processed, y_train, X_test_processed, y_test)
            model_performance_description = "KNN was chosein since we identified some nonlinear relationships in our data and since it can handle both categorical and continuous predictors. One of the the cons we found was an even lower r2 score before reducing the features of our model."
        elif chosen_model == "Random Forest":
            metrics, y_test_predictions = rf_model(X_train, y_train, X_test, y_test)
            model_performance_description = "Given the overfitting present in the Decision Tree model before feature reduction, Random Forest was chosen since one of it's strengths is to reduce overfitting and improve generalization, recursive feature elimination CV was integrated to this model to help reduce overfitting even more and start to identify relevant variables. We found this model to be computationally expensive, since it took the longest to fit and the results we're not extraordinary."
        elif chosen_model == "Gradient Boosting":
            metrics, y_test_predictions, feature_importance, sorted_idx, estimators, feature_names = gb_model(X_train, y_train, X_test, y_test)    
        model_performance_description = "Given the low results achieved in the rest of the models, Gradient Boosting was chosen because of it's method of combining weak models to achieve a strong predictor. The model seems generalizes well also to new and unseen data, as seen in test and validation scores. RFECV was applied to the final model, which helped us to identify and remove the weakest features to optimize our model."

        st.write("Below we display the model performance on the train and test set as measured by multiple evaluation metrics and provide a brief description on each model.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(metrics)
        with col2:
            st.write("Model Performance Description:")
            st.write(model_performance_description)
        
        st.write("Below you can see the actual number of total users compared to the predicted number of total users using this model for the unseen test data")

        predictions_df = create_predictions_df(y_test_predictions, test_dates, y_test)

        # add calendar for start and end dates
        date_col_1, date_col_2 = st.columns(2)
        
        with date_col_1:
            start_date = st.date_input("Start date", value=predictions_df["datetime"].min(), min_value=predictions_df["datetime"].min(),max_value=predictions_df["datetime"].max())
        with date_col_2:
            end_date = st.date_input("End date", value=predictions_df["datetime"].max(), min_value=predictions_df["datetime"].min(), max_value=predictions_df["datetime"].max())

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # filter data based on selected dates
        partiv_df = predictions_df[(predictions_df["datetime"] >= start_date) & (predictions_df["datetime"] <= end_date)]
        partiv_df["hour"] = partiv_df["datetime"]
        partiv_df['month'] = partiv_df['datetime'].dt.month
        partiv_df['day'] = partiv_df['datetime'].dt.date

        # title for time series graph
        st.subheader("Actual vs Predicted Bike Rentals")

        # select granularity of time series plot
        selected_timeframe = st.radio("", ["hour", "day", "month"], horizontal=True)

        #group df by chosen granularity
        df_grouped = partiv_df.groupby(selected_timeframe).sum().reset_index()

        # create figure with total number of users
        fig = px.line()

        # add trace for actual count
        actual_trace = go.Scatter(x=df_grouped[selected_timeframe], y=df_grouped["actual_count"], name='Actual Users')
        fig.add_trace(actual_trace)

        # add trace for predicted count
        predicted_trace = go.Scatter(x=df_grouped[selected_timeframe], y=df_grouped["predicted_count"], name='Predicted Users')
        fig.add_trace(predicted_trace)

        #configure figure layout
        fig.update_yaxes(title="Total Bikes Rented")
        fig.update_xaxes(title=selected_timeframe)
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, range=[0, None]))
        fig.update_layout(legend=dict(title='Type of User'))

        # show the plot
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
    
        # final model
        st.subheader("Final Model")
        st.markdown('''
        As mentioned in the 'Models" tab, after trying various models, we decided to use **Gradient Boosting** as our final model as it provided the best performance (highest training and validation scores and less overfitting on the training data).
        ''')

        st.markdown('''
        For this final model, we used Revursive Feature Elimination (RFECV), to select features based on feature importance. The following features were excluded from the model given the results of RFECV and feature importances: 
        - `mnth_start` 
        - `mnth_end` 
        - `quarter` 
        - `ytd_mean_daily_temp` 
        - `ytd_mean_daily_hum` 
        - `ytd_min_daily_temp` 
        - `ytd_min_daily_hum`
        - `windspeed`
        - `is_night`
        
        The code used for the RFECV can be found below.
        ''')
        
        st.code('''
        # create an RFECV object
        rfecv_gb = RFECV(estimator=grid_search_gb.best_estimator_, step=1, cv=5, scoring='r2', n_jobs=-1)

        # fit the GridSearchCV object to the training data
        rfecv_gb.fit(X_train, y_train)
        
        # print selected features
        print("Selected features: ", X_train.columns[rfecv.support_])
        ''')

        st.write("You can find the hyperparameters of the final model, obtained via gridsearch, below.")
        st.code('''
        GradientBoostingRegressor(
        max_depth = 5, # maximum depth of each regression estimator
        max_features = None, # maximum features to consider when looking for best split
        min_samples_leaf = 1, # minimum samples required to be at a leaf node
        min_samples_split = 7, # minimum number of samples required to split an internal node
        alpha = 0.1, # regularization parameter
        n_estimators = 150 # number of boosting stages
        )
        ''')

        st.write("You can find the top 10 most important features in our model below.")

        # call gb_model function to get feature_importance
        metrics_gb, y_test_predictions, feature_importance, sorted_idx, estimators, feature_names = gb_model(X_train, y_train, X_test, y_test)

        top_k=10
        feature_importance_topk = feature_importance[sorted_idx][-top_k:]
        feature_names_topk = X_train.columns[sorted_idx][-top_k:]

        importances_df = pd.DataFrame({'Feature': feature_names_topk, 'Importance': feature_importance_topk})
        importances_df = importances_df.sort_values("Importance", ascending=False)

        fig = go.Figure(go.Bar(x=importances_df["Feature"], y=importances_df["Importance"]))

        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title='Importance',
            width=400,
            height=200,
            margin=dict(l=50, r=50, t=30, b=20),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, range=[0, None]), height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("`is_rush_hour` was the most important feature in the model, followed by `day_period` and then the cosine of the hour `cos_hour`")

        #plot model predictions vs residuals
        st.write("Below you can find a plots showing the relationship between predicted values and actual values (left) as well as predicted values and their residuals (right).")
            
        col_1, col_2 = st.columns(2)
        
        with col_1:
            #plot model predictions vs residuals
            residuals = y_test - y_test_predictions
            residuals_df = pd.DataFrame({'predicted': y_test_predictions, 
                                         'residuals': residuals,
                                         'actual': y_test})

            fig = px.scatter(residuals_df, x='predicted', y='residuals', title='Predictions vs. Residuals')
            fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Residuals do not appear entirely normally distributed but are centered around 0 with some outliers.")
        
        with col_2:
            #plot model predictions vs actual
            fig = px.scatter(residuals_df, x='predicted', y='actual', title='Predictions vs. Actual')
            fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("There is a strong positive linear relationship between actual and predicted values as expected given the high model performance")

        # plot an example estimator
        st.write("You can see an example estimator from our model below.")

        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(90,20))
        tree = plot_tree(estimators[2][0], filled=True, 
        rounded=True, fontsize=30,feature_names=feature_names,
        impurity=False, proportion=False, precision=0, node_ids=False, label=None, class_names=None, max_depth=None, ax=None)
        
        st.pyplot()
    
############### PART III ###############    
def partv_page():
    # title
    st.markdown("<h2 style='text-align: center;'>Part V: Model Predictions</h2>", unsafe_allow_html=True)
    
    # description of how to use page
    st.write("Make your own prediction using the selectors below! Choose a value for each feature and see the corresponsing predicted number of bikes according to the final model.")
        
    # Time inputs
    st.write("Time inputs")
    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        year = st.selectbox("Year", [2011, 2012])
    with col_2:
        month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    with col_3:
        weekday = st.selectbox("Weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    with col_4:
        hour = st.slider("Hour", 0, 23, 11, step=1)
    
    # Temperature  inputs
    st.write("Today's Temperature Inputs (normalized)")
    col_1, col_2, col_3, col_4, col_5 = st.columns(5)
    with col_1:
        temp = st.slider("Temperature", 0.0, 1.0, 0.5, step=0.01)
    with col_2:
        atemp = st.slider("Feels Like Temperature", 0.0, 1.0, 0.5, step=0.01)
    with col_3:
        min_daily_temp = st.slider("Minimum Temperature", 0.0, 1.0, 0.5, step=0.01)
    with col_4:
        mean_daily_temp = st.slider("Average Temperature", 0.0, 1.0, 0.5, step=0.01)   
    with col_5:
        max_daily_temp = st.slider("Maximum Temperature", 0.0, 1.0, 0.5, step=0.01)

    # humidity inputs
    st.write("Today's Humidity Inputs (normalized)")
    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        hum = st.slider("Humidity", 0.0, 1.0, 0.5, step=0.01)
    with col_2:
        min_daily_hum = st.slider("Minimum Humidity", 0.0, 1.0, 0.5, step=0.01)
    with col_3:
        mean_daily_hum = st.slider("Average Humidity", 0.0, 1.0, 0.5, step=0.01)
    with col_4:
        st.write("")
        
    # windspeed and selection box for weather conditions inputs
    st.write("Weather Inputs")
    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        windspeed = st.slider("Windspeed", 0.0, 1.0, 0.1, step=0.01)
    with col_2:
        weather = st.selectbox("Weather conditions", ["Clear, Few clouds, Partly cloudy, Partly cloudy", "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist", "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"]) 
    with col_3:
        st.write("")
    with col_4:
        st.write("")
    
    # recode selected weather condition
    if weather == "Clear, Few clouds, Partly cloudy, Partly cloudy":
        weathersit = 1
    elif weather == "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist":
        weathersit = 2
    elif weather == "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds":
        weathersit = 3
    elif weather == "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog":
        weathersit = 4

    # define weather factor based on inputted weathersit and windspeed
    weather_factor = weathersit * windspeed

    # recode selected month & define season based on month input
    if month == "January":
        month_coded = 1
        season = 1
    elif month == "February":
        month_coded = 2
        season = 1
    elif month == "March":
        month_coded = 3
        season = 1
    elif month == "April":
        month_coded = 4
        season = 2
    elif month == "May":
        month_coded = 5
        season = 2
    elif month == "June":
        month_coded = 6
        season = 2
    elif month == "July":
        month_coded = 7
        season = 3
    elif month == "August":
        month_coded = 8
        season = 3
    elif month == "September":
        month_coded = 9
        season = 3
    elif month == "October":
        month_coded = 10
        season = 4
    elif month == "November":
        month_coded = 11
        season = 4
    elif month == "December":
        month_coded = 12
        season = 4
    
    # define cos_month based on month input
    cos_month = np.cos(2*np.pi*month_coded/12)
    
    # define cos_hour based on hour input
    cos_hour = np.cos(2*np.pi*hour/12)
    
    # recode weekday and define is_weekend based on weekday
    if weekday == "Monday":
        weekday_coded = 1
        is_weekend = 0
    elif weekday == "Tuesday":
        weekday_coded = 2
        is_weekend = 0
    elif weekday == "Wednesday":
        weekday_coded = 3
        is_weekend = 0
    elif weekday == "Thursday":
        weekday_coded = 4
        is_weekend = 0
    elif weekday == "Friday":
        weekday_coded = 5
        is_weekend = 0
    elif weekday == "Saturday":
        weekday_coded = 6
        is_weekend = 1
    elif weekday == "Sunday":
        weekday_coded = 0
        is_weekend = 1
    
    # define cos_weekday based on week_day input
    cos_weekday = np.cos(2*np.pi*weekday_coded/7)
    
    # define day_period
    if hour <=5:
        day_period = 1
    elif 6<= hour <= 11:
        day_period = 2
    elif 12<= hour <= 17:
        day_period = 3
    elif 18<= hour <= 23:
        day_period = 4
        
    # recode selected rush hour status
    if hour in [8, 17, 18]:
        is_rush_hour = 1
    else:
        is_rush_hour = 0

    # recode selected year
    if year == 2012:
        yr = 1
    elif year == 2011:
        yr = 0
        
    # create dataframe with selected values for each feature
    input_x_test = pd.DataFrame(
             {'atemp': atemp,
              'cos_hour': cos_hour,
              'cos_month': cos_month,
              'cos_weekday': cos_weekday,
              'day_period': day_period,
              'hum': hum,
              'is_rush_hour': is_rush_hour,
              'is_weekend': is_weekend,
              'max_daily_temp': max_daily_temp,
              'mean_daily_hum': mean_daily_hum,
              'mean_daily_temp': mean_daily_temp,
              'min_daily_hum': min_daily_hum,
              'min_daily_temp': min_daily_temp,
              'season': season,
              'temp': temp,
              'weather_factor': weather_factor,
              'weathersit': weathersit,
              'yr': yr
        }, index = [0])

    # import preprocessed training and testing datasets
    X_train, y_train, X_test, y_test, X_train_processed, X_test_processed, test_dates = preprocess()

    # train model on x_train and y_train and predict using user inputs 
    metrics_gb, prediction_from_input, feature_importance, sorted_idx, estimators, feature_names = gb_model(X_train, y_train, input_x_test, [0])
    # display prediction
    st.markdown("<h3 style='text-align: left;'>Predicted Numer of Bike Rentals:</h3>", unsafe_allow_html=True)
    
    if prediction_from_input < 0:
        final_prediction = 0
    else:
        final_prediction = prediction_from_input.astype(int)
    
    st.metric("", final_prediction)

 ##############################

#Define a dictionary with the page names and their respective functions
pages = {
    "Part I: Introduction": parti_page,
    "Part II: Basic Exploratory Analysis": partii_page,
    "Part III: Customer Behavior Analysis": partiii_page,
    "Part IV: Model Building": partiv_page,
    "Part V: Model Predictions": partv_page
}

# Add a sidebar menu to select the page
selected_page = st.sidebar.radio("Table of Contents", list(pages.keys()))

# Display the selected page
pages[selected_page]()

# if __name__ == "__main__":

#     # This is to configure some aspects of the app
#     st.set_page_config(
#         layout="wide", page_title="Madrid Mobility Dashboard", page_icon=":car:"
#     )

#     # Write titles in the main frame and the side bar
#     st.title("Madrid Mobility Dashboard")
#     st.sidebar.title("Options")

#     # Call main function
#     main()
