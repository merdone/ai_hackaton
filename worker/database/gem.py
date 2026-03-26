import random
from datetime import datetime, timedelta
# Импортируем готовый инстанс БД и класс из твоего файла
from db import db 

def generate_synthetic_shift(start_time_str="2026-03-25T08:00:00", hours=8):
    """Генерирует полную смену работы терминала и записывает в БД"""
    
    # 1. Настройка параметров симуляции
    shift_start = datetime.fromisoformat(start_time_str)
    shift_end = shift_start + timedelta(hours=hours)
    
    # Справочники
    zones = ["Зона вивантаження", "Зона сортування", "Зона пакування", "Транзитний коридор", "Зона відпочинку"]
    tasks = {
        "Зона вивантаження": ["розвантаження", "переміщення вантажу", "очікування / простій"],
        "Зона сортування": ["сортування відправлень", "очікування / простій"],
        "Зона пакування": ["перепакування", "маркування", "очікування / простій"],
        "Транзитний коридор": ["переміщення між зонами", "переміщення вантажу"],
        "Зона відпочинку": ["очікування / простій"]
    }
    
    # Профили рабочих (W-003 - проблемный, у него другие вероятности)
    workers = ["W-001", "W-002", "W-003", "W-004"]
    
    print(f"Починаємо генерацію даних з {shift_start} до {shift_end}...")
    
    # Убедимся, что таблицы созданы
    db.init_schema()
    
    events_created = 0

    # 2. Генерируем непрерывный поток событий для каждого рабочего
    for worker_id in workers:
        current_time = shift_start
        current_zone = random.choice(zones)
        
        while current_time < shift_end:
            # Логика перемещений: с вероятностью 20% рабочий меняет зону
            if random.random() < 0.2:
                current_zone = random.choice(zones)
                action = "переміщення між зонами"
                # Перемещение занимает от 30 секунд до 2 минут
                duration_sec = random.randint(30, 120) 
            else:
                # Рабочий делает что-то в текущей зоне
                possible_actions = tasks[current_zone]
                
                # ХАК ДЛЯ ПРЕЗЕНТАЦИИ: Рабочий W-003 халтурит
                if worker_id == "W-003" and "очікування / простій" in possible_actions:
                    # 60% шанс, что он будет простаивать
                    action = random.choices(
                        possible_actions, 
                        weights=[0.1 if a != "очікування / простій" else 0.6 for a in possible_actions]
                    )[0]
                else:
                    # Нормальные рабочие простаивают редко (5% шанс)
                    action = random.choices(
                        possible_actions, 
                        weights=[0.9 if a != "очікування / простій" else 0.05 for a in possible_actions]
                    )[0]
                
                # Длительность обычного действия: от 2 до 15 минут
                duration_sec = random.randint(120, 900)

            # Ограничиваем время концом смены
            end_time = current_time + timedelta(seconds=duration_sec)
            if end_time > shift_end:
                end_time = shift_end
                duration_sec = (end_time - current_time).total_seconds()

            # Имитация AI-модели: уверенность от 70% до 99%
            confidence = round(random.uniform(0.70, 0.99), 2)
            
            # Запись в БД
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
            current_time = end_time # Следующее действие начинается ровно в момент окончания предыдущего

    print(f"Готово! Згенеровано та записано {events_created} подій.")

if __name__ == "__main__":
    generate_synthetic_shift()