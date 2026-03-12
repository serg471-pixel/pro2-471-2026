import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime, timedelta


st.set_page_config(page_title="Прогноз опадів ML", page_icon="🌦")



def fetch_weather_data(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "precipitation_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "windspeed_10m_max",
            "relative_humidity_2m_max"
        ],
        "timezone": "Europe/Kyiv"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['daily'])


        df['is_precip'] = (df['precipitation_sum'] > 0).astype(int)
        return df
    except Exception as e:
        st.error(f"Помилка при отриманні даних: {e}")
        return None



def process_and_train(df):

    df['target_next_day'] = df['is_precip'].shift(-1)


    data_model = df.dropna().copy()


    features = ['temperature_2m_max', 'temperature_2m_min', 'windspeed_10m_max']
    X = data_model[features]
    y = data_model['target_next_day']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, acc, report, features


# --- 3. Інтерфейс Streamlit ---
st.title("🌦 ML Прогноз опадів (Open-Meteo)")
st.markdown("""
Цей застосунок завантажує історичні дані погоди, навчає модель Random Forest 
та прогнозує ймовірність опадів на основі метеорологічних показників.
""")

with st.sidebar:
    st.header("Налаштування")
    lat = st.number_input("Широта (Latitude)", value=50.45)  # Київ
    lon = st.number_input("Довгота (Longitude)", value=30.52)


    end_dt = datetime.now() - timedelta(days=2)
    start_dt = end_dt - timedelta(days=365)

    st.info(f"Дані будуть взяті за період:\n{start_dt.date()} — {end_dt.date()}")


if st.button("🚀 Отримати дані та зробити прогноз"):
    with st.spinner('Працюю з даними...'):

        df_raw = fetch_weather_data(lat, lon, start_dt.date().isoformat(), end_dt.date().isoformat())

        if df_raw is not None:
            st.success(f"Завантажено {len(df_raw)} днів спостережень.")


            df_raw.to_csv("weather_daily.csv", index=False)
            st.caption("Дані збережено у файл `weather_daily.csv`")


            model, acc, report, feat_cols = process_and_train(df_raw)


            col1, col2 = st.columns(2)
            col1.metric("Точність моделі (Accuracy)", f"{acc:.2%}")
            col2.metric("Днів з опадами у вибірці", f"{df_raw['is_precip'].sum()}")

            with st.expander("Повний звіт класифікації"):
                st.json(report)


            st.divider()
            st.subheader("🔮 Результат прогнозу")


            latest_features = df_raw[feat_cols].tail(1)
            prediction = model.predict(latest_features)[0]
            probability = model.predict_proba(latest_features)[0]  # [P(0), P(1)]

            prob_rain = probability[1]

            if prediction == 1 or prob_rain > 0.4:
                st.error(f"### ☔ Очікуються опади")
                st.write(f"Ймовірність: **{prob_rain:.1%}**")
            else:
                st.success(f"### ☀️ Опадів не очікується")
                st.write(f"Ймовірність опадів лише **{prob_rain:.1%}**")


            st.write("---")
            st.write("**Вплив факторів на прогноз:**")
            importance = pd.DataFrame({'Ознака': feat_cols, 'Важливість': model.feature_importances_})
            st.bar_chart(importance.set_index('Ознака'))

else:
    st.warning("Натисніть кнопку вище, щоб активувати модель.")