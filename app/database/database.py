import sqlite3
import json

from app.settings import get_app_settings


class LogisticsDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = None
        self.connect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        if not self.connection:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row

    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute_query(self, query, params=None, fetch=True):
        if not self.connection:
            raise Exception("Database connection is not established.")

        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.connection.commit()
            return [dict(row) for row in cursor.fetchall()] if fetch else None
        except sqlite3.Error as e:
            self.connection.rollback()
            print(f"Database error: {e}")
            raise

    def init_schema(self):
        """Creates the operations_log table if it doesn't exist"""
        query = """
        CREATE TABLE IF NOT EXISTS operations_log (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT,
            task_classification TEXT,
            zone TEXT,
            timestamp_start DATETIME,
            timestamp_end DATETIME,
            duration REAL,
            confidence_score REAL,
            source_id TEXT,
            metadata TEXT
        );
        """
        self.execute_query(query, fetch=False)

    def log_event(self, worker_id, classification, zone, start_time, end_time, confidence, source_id="camera_1",
                  metadata=None):
        """Logs a digital event into the database"""
        query = """
        INSERT INTO operations_log 
        (worker_id, task_classification, zone, timestamp_start, timestamp_end, duration, confidence_score, source_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        duration = (end_time - start_time).total_seconds()
        meta_json = json.dumps(metadata) if metadata else "{}"

        params = (
            worker_id, classification, zone,
            start_time.isoformat(), end_time.isoformat(),
            duration, confidence, source_id, meta_json
        )
        self.execute_query(query, params, fetch=False)

    def get_zone_analytics(self):
        """Example of data aggregation for dashboard (process metrics)"""
        query = """
        SELECT 
            zone,
            task_classification,
            COUNT(*) as event_count,
            SUM(duration) as total_duration_sec,
            AVG(confidence_score) as avg_confidence
        FROM operations_log
        GROUP BY zone, task_classification
        """
        return self.execute_query(query)

    def get_worker_history(self, worker_id, start_time=None, end_time=None):
        """
        Gets the event history for a specific worker, optionally filtered by a time range.
        """
        query = "SELECT * FROM operations_log WHERE worker_id = ?"
        params = [worker_id]

        if start_time:
            query += " AND timestamp_start >= ?"
            params.append(start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time)
        if end_time:
            query += " AND timestamp_end <= ?"
            params.append(end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time)

        query += " ORDER BY timestamp_start DESC"

        return self.execute_query(query, tuple(params))

    def get_events_by_time_range(self, start_time, end_time):
        query = """
        SELECT * FROM operations_log 
        WHERE timestamp_start >= ? AND timestamp_end <= ?
        ORDER BY timestamp_start ASC
        """

        st_formatted = start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time
        et_formatted = end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time

        return self.execute_query(query, (st_formatted, et_formatted))

    def get_worker_efficiency(self, worker_id, start_time, end_time):
        query = """
        SELECT 
            task_classification,
            COUNT(*) as tasks_completed,
            SUM(duration) as total_time_sec,
            AVG(confidence_score) as avg_ai_confidence
        FROM operations_log
        WHERE worker_id = ? AND timestamp_start >= ? AND timestamp_end <= ?
        GROUP BY task_classification
        """

        st_formatted = start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time
        et_formatted = end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time

        return self.execute_query(query, (worker_id, st_formatted, et_formatted))


db = LogisticsDatabase(get_app_settings().database_path)