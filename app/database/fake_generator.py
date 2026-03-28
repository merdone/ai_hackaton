import random
from datetime import datetime, timedelta
# Імпортуємо готовий інстанс БД і клас із вашого файла
from database import db

def generate_synthetic_shift(start_time_str="2026-03-25T08:00:00", hours=8):
    """Генерує повну зміну роботи термінала та записує події в БД"""

    # 1. Налаштування параметрів симуляції
    shift_start = datetime.fromisoformat(start_time_str)
    shift_end = shift_start + timedelta(hours=hours)
    
    # Довідники
    zones = ["Зона вивантаження", "Зона сортування", "Зона пакування", "Транзитний коридор", "Зона відпочинку"]
    tasks = {
        "Зона вивантаження": ["розвантаження", "переміщення вантажу", "очікування / простій"],
        "Зона сортування": ["сортування відправлень", "очікування / простій"],
        "Зона пакування": ["перепакування", "маркування", "очікування / простій"],
        "Транзитний коридор": ["переміщення між зонами", "переміщення вантажу"],
        "Зона відпочинку": ["очікування / простій"]
    }
    
    # Профілі працівників (W-003 - проблемний, у нього інші ймовірності)
    workers = ["W-001", "W-002", "W-003", "W-004"]
    
    print(f"Починаємо генерацію даних з {shift_start} до {shift_end}...")
    
    # Переконаймося, що таблиці створені
    db.init_schema()
    
    events_created = 0

    # 2. Генеруємо безперервний потік подій для кожного працівника
    for worker_id in workers:
        current_time = shift_start
        current_zone = random.choice(zones)
        
        while current_time < shift_end:
            # Логіка переміщень: з імовірністю 20% працівник змінює зону
            if random.random() < 0.2:
                current_zone = random.choice(zones)
                action = "переміщення між зонами"
                # Переміщення триває від 30 секунд до 2 хвилин
                duration_sec = random.randint(30, 120)
            else:
                # Працівник виконує дію у поточній зоні
                possible_actions = tasks[current_zone]
                
                # ХАК ДЛЯ ПРЕЗЕНТАЦІЇ: працівник W-003 халтурить
                if worker_id == "W-003" and "очікування / простій" in possible_actions:
                    # 60% імовірність, що він простоюватиме
                    action = random.choices(
                        possible_actions, 
                        weights=[0.1 if a != "очікування / простій" else 0.6 for a in possible_actions]
                    )[0]
                else:
                    # Звичайні працівники простоюють рідко (5% імовірність)
                    action = random.choices(
                        possible_actions, 
                        weights=[0.9 if a != "очікування / простій" else 0.05 for a in possible_actions]
                    )[0]
                
                # Тривалість звичайної дії: від 2 до 15 хвилин
                duration_sec = random.randint(120, 900)

            # Обмежуємо час кінцем зміни
            end_time = current_time + timedelta(seconds=duration_sec)
            if end_time > shift_end:
                end_time = shift_end
                duration_sec = (end_time - current_time).total_seconds()

            # Імітація AI-моделі: впевненість від 70% до 99%
            confidence = round(random.uniform(0.70, 0.99), 2)
            
            # Запис у БД
            db.log_event(
                worker_id=worker_id,
                classification=action,
                zone=current_zone,
                start_time=current_time,
                end_time=end_time,
                confidence=confidence,
                source_id=f"camera_{zones.index(current_zone) + 1}"
            )
            
            events_created += 1
            current_time = end_time # Наступна дія починається рівно в момент завершення попередньої

    print(f"Готово! Згенеровано та записано {events_created} подій.")

if __name__ == "__main__":
    generate_synthetic_shift()