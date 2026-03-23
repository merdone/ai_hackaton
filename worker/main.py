import sqlite3
import time
import random
from datetime import datetime

# Подключаемся к базе. Так как мы прокинули volume, файл создастся в папке data/
conn = sqlite3.connect('/app/data/events.db')
cursor = conn.cursor()

# Создаем таблицу, если ее нет
cursor.execute('''
    CREATE TABLE IF NOT EXISTS events (
        timestamp TEXT,
        worker_id TEXT,
        action TEXT,
        zone TEXT,
        confidence REAL
    )
''')
conn.commit()

print("Worker запущен. Начинаю генерацию тестовых событий...")

# Вечный цикл, имитирующий работу CV [cite: 10, 26]
while True:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    worker_id = random.choice(["ID_1", "ID_2", "ID_3"])
    action = random.choice(["Moving", "Sorting", "Idle"])
    zone = random.choice(["Zone_A", "Zone_B"])
    confidence = round(random.uniform(0.65, 0.99), 2)

    # Имитация записи события при confidence > 0.6 [cite: 12, 13]
    cursor.execute('INSERT INTO events VALUES (?, ?, ?, ?, ?)',
                   (now, worker_id, action, zone, confidence))
    conn.commit()

    print(f"Событие записано: {now} | {worker_id} | {action}")
    time.sleep(2)  # Имитация задержки обработки кадра