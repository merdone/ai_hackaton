import streamlit as st
import sqlite3
import pandas as pd
import time

st.set_page_config(page_title="Склад Аналитика", layout="wide")

st.title("🚧 Streamlit + SQLite: Hello World 🚧")

# Читаем из той же базы
conn = sqlite3.connect('/app/data/events.db')

# Простая кнопка для ручного обновления (позже заменим на st_autorefresh)
if st.button("Обновить данные из БД"):
    try:
        # SELECT запрос к событиям
        df = pd.read_sql_query("SELECT * FROM events ORDER BY timestamp DESC LIMIT 10", conn)

        st.write("Последние 10 событий, которые сгенерировал Worker:")
        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.warning(f"Ждем, пока Worker создаст базу... ({e})")