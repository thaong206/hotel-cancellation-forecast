
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Evaluation metrics
def evaluate_forecast(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'MSE': mse}

st.title("Hotel Booking Cancellation Forecast")

uploaded_file = st.file_uploader("Upload hotel_bookings.csv", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df['date'] = pd.to_datetime(df[['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']].astype(str).agg('-'.join, axis=1), errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    df = df.drop(columns=['agent', 'company', 'reservation_status_date', 'adr', 'required_car_parking_spaces',
                          'total_of_special_requests', 'arrival_date_week_number', 'booking_changes',
                          'assigned_room_type', 'reservation_status', 'distribution_channel'], errors='ignore')
    df = df.dropna()

    st.subheader("Exploratory Data Analysis")

    st.markdown("### Daily Cancellations Over Time")
    df_daily = df.resample('D').sum()
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    df_daily['is_canceled'].plot(ax=ax1)
    ax1.set_title("Daily Cancellations Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Number of Cancellations")
    ax1.grid(True)
    st.pyplot(fig1)

    st.markdown("### Cancellations Distribution")
    fig2, ax2 = plt.subplots()
    sns.barplot(x=df["is_canceled"].value_counts().index, y=df["is_canceled"].value_counts().values, ax=ax2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Not Canceled", "Canceled"])
    ax2.set_title("Hotel Booking Cancellations")
    ax2.set_ylabel("Number of Bookings")
    st.pyplot(fig2)

    st.markdown("### Monthly Cancellation Trends")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x="arrival_date_month", hue="is_canceled", 
                  order=["January", "February", "March", "April", "May", "June",
                         "July", "August", "September", "October", "November", "December"], ax=ax3)
    ax3.set_title("Monthly Cancellation Trends")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    st.pyplot(fig3)

    st.markdown("### Feature Correlation Heatmap")
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        df_encoded[col] = df_encoded[col].factorize()[0]
    fig4, ax4 = plt.subplots(figsize=(16, 10))
    sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
    ax4.set_title("Feature Correlation Heatmap", fontsize=14)
    st.pyplot(fig4)

    # Monthly cancellation rate
    monthly_cancellations = df.groupby(["arrival_date_year", "arrival_date_month"])["is_canceled"].agg(["sum", "count"]).reset_index()
    monthly_cancellations["cancellation_rate"] = monthly_cancellations["sum"] / monthly_cancellations["count"]
    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    monthly_cancellations["arrival_date_month"] = pd.Categorical(monthly_cancellations["arrival_date_month"], categories=month_order, ordered=True)
    monthly_cancellations = monthly_cancellations.sort_values(["arrival_date_year", "arrival_date_month"])

    st.markdown("### Monthly Cancellation Rate Over Time")
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=monthly_cancellations, x="arrival_date_month", y="cancellation_rate", hue="arrival_date_year", marker="o", ax=ax5)
    ax5.set_title("Hotel Booking Cancellation Rate Over Time")
    ax5.set_xlabel("Month")
    ax5.set_ylabel("Cancellation Rate")
    ax5.legend(title="Year")
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
    ax5.grid(True)
    st.pyplot(fig5)

    # Convert to time series
    date_range = pd.date_range(start="2015-07", periods=len(monthly_cancellations), freq="M")
    ts = pd.Series(monthly_cancellations["cancellation_rate"].values, index=date_range, name="cancellation_rate")

    st.subheader("Seasonal Decomposition")
    decomposition_type = st.selectbox("Choose decomposition model", ["additive", "multiplicative"])
    decomposition = seasonal_decompose(ts, model=decomposition_type, period=12)

    fig6, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    decomposition.observed.plot(ax=axes[0], title="Observed")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
    decomposition.resid.plot(ax=axes[3], title="Residuals")
    st.pyplot(fig6)

    st.subheader("Forecasting Model")
    selected_model = st.selectbox("Select a forecasting model", ["ARIMA", "Exponential Smoothing", "Prophet"])
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    arima_model = ARIMA(train, order=(1, 1, 1)).fit()
    arima_forecast = arima_model.forecast(steps=len(test))
    arima_metrics = evaluate_forecast(test, arima_forecast)

    ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=6).fit()
    ets_forecast = ets_model.forecast(steps=len(test))
    ets_metrics = evaluate_forecast(test, ets_forecast)

    prophet_df = ts.reset_index().rename(columns={"index": "ds", "cancellation_rate": "y"})
    prophet_train = prophet_df.iloc[:train_size]
    prophet_model = Prophet()
    prophet_model.fit(prophet_train)
    future = prophet_model.make_future_dataframe(periods=len(test), freq='M')
    prophet_forecast_df = prophet_model.predict(future)
    prophet_forecast = prophet_forecast_df.iloc[-len(test):]["yhat"].values
    prophet_metrics = evaluate_forecast(test.values, prophet_forecast)

    evaluation_df = pd.DataFrame({
        "Model": ["ARIMA", "Exponential Smoothing", "Prophet"],
        "RMSE": [arima_metrics["RMSE"], ets_metrics["RMSE"], prophet_metrics["RMSE"]],
        "MAE": [arima_metrics["MAE"], ets_metrics["MAE"], prophet_metrics["MAE"]],
        "MAPE": [arima_metrics["MAPE"], ets_metrics["MAPE"], prophet_metrics["MAPE"]],
        "MSE": [arima_metrics["MSE"], ets_metrics["MSE"], prophet_metrics["MSE"]]
    })

    st.write(f"**Selected Model:** {selected_model}")
    if selected_model == 'ARIMA':
        evaluation_df = evaluation_df[evaluation_df['Model'] == 'ARIMA']
        forecast_plot = arima_forecast
    elif selected_model == 'Exponential Smoothing':
        evaluation_df = evaluation_df[evaluation_df['Model'] == 'Exponential Smoothing']
        forecast_plot = ets_forecast
    else:
        evaluation_df = evaluation_df[evaluation_df['Model'] == 'Prophet']
        forecast_plot = prophet_forecast

    st.dataframe(evaluation_df)

    st.subheader("Forecast vs Actual")
    fig7, ax7 = plt.subplots()
    test.plot(ax=ax7, label="Actual", marker='o')
    pd.Series(forecast_plot, index=test.index).plot(ax=ax7, label=selected_model, linestyle="--")
    plt.legend()
    st.pyplot(fig7)
