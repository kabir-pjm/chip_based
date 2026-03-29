"""
Thermal Database — SQLite Logging & Replay System
===================================================
Stores all thermal readings, anomaly events, and governor actions
for post-mortem analysis, dashboarding, and replay.
"""

import sqlite3
import time
from typing import List, Dict, Optional
from pathlib import Path


class ThermalDatabase:
    """
    Persistent storage for thermal telemetry using SQLite.
    Supports high-frequency writes (~2/sec) and analytical queries.
    """

    def __init__(self, db_path: str = "data/thermal_log.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent reads
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                zone TEXT NOT NULL,
                temperature REAL NOT NULL,
                slope REAL,
                stat_score REAL,
                ml_score REAL,
                fused_score REAL,
                state TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(timestamp);
            CREATE INDEX IF NOT EXISTS idx_readings_zone ON readings(zone);
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
        """)
        self.conn.commit()

    def log_reading(self, zone: str, temperature: float, slope: float,
                    stat_score: float, ml_score: float, fused_score: float,
                    state: str):
        """Log a single thermal reading."""
        self.conn.execute(
            "INSERT INTO readings (timestamp, zone, temperature, slope, "
            "stat_score, ml_score, fused_score, state) VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), zone, temperature, slope, stat_score, ml_score,
             fused_score, state)
        )
        self.conn.commit()

    def log_event(self, event_type: str, details: str = ""):
        """Log a governor event (throttle, resume, warning)."""
        self.conn.execute(
            "INSERT INTO events (timestamp, event_type, details) VALUES (?,?,?)",
            (time.time(), event_type, details)
        )
        self.conn.commit()

    def get_readings(self, zone: Optional[str] = None,
                     last_n: int = 100) -> List[Dict]:
        """Retrieve recent readings, optionally filtered by zone."""
        if zone:
            rows = self.conn.execute(
                "SELECT * FROM readings WHERE zone=? ORDER BY timestamp DESC LIMIT ?",
                (zone, last_n)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM readings ORDER BY timestamp DESC LIMIT ?",
                (last_n,)
            ).fetchall()

        cols = ['id', 'timestamp', 'zone', 'temperature', 'slope',
                'stat_score', 'ml_score', 'fused_score', 'state']
        return [dict(zip(cols, row)) for row in rows]

    def get_events(self, last_n: int = 50) -> List[Dict]:
        """Retrieve recent events."""
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
            (last_n,)
        ).fetchall()
        cols = ['id', 'timestamp', 'event_type', 'details']
        return [dict(zip(cols, row)) for row in rows]

    def get_zone_stats(self, zone: str, window_seconds: int = 300) -> Dict:
        """Get aggregated statistics for a zone over a time window."""
        cutoff = time.time() - window_seconds
        row = self.conn.execute(
            """SELECT
                AVG(temperature) as avg_temp,
                MAX(temperature) as max_temp,
                MIN(temperature) as min_temp,
                AVG(slope) as avg_slope,
                COUNT(*) as sample_count,
                SUM(CASE WHEN state != 'MONITORING' THEN 1 ELSE 0 END) as anomaly_count
            FROM readings
            WHERE zone=? AND timestamp > ?""",
            (zone, cutoff)
        ).fetchone()

        return {
            'avg_temp': round(row[0] or 0, 2),
            'max_temp': round(row[1] or 0, 2),
            'min_temp': round(row[2] or 0, 2),
            'avg_slope': round(row[3] or 0, 4),
            'sample_count': row[4] or 0,
            'anomaly_count': row[5] or 0,
        }

    def export_csv(self, filepath: str = "data/thermal_export.csv"):
        """Export all readings to CSV for external analysis."""
        import csv
        rows = self.conn.execute(
            "SELECT * FROM readings ORDER BY timestamp"
        ).fetchall()
        cols = ['id', 'timestamp', 'zone', 'temperature', 'slope',
                'stat_score', 'ml_score', 'fused_score', 'state']
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        return filepath

    def replay_generator(self, speed: float = 1.0):
        """
        Yields readings in chronological order for replay.
        Speed multiplier: 1.0 = real-time, 2.0 = double speed.
        """
        rows = self.conn.execute(
            "SELECT * FROM readings ORDER BY timestamp"
        ).fetchall()
        cols = ['id', 'timestamp', 'zone', 'temperature', 'slope',
                'stat_score', 'ml_score', 'fused_score', 'state']

        prev_ts = None
        for row in rows:
            reading = dict(zip(cols, row))
            if prev_ts is not None:
                delay = (reading['timestamp'] - prev_ts) / speed
                if delay > 0:
                    time.sleep(delay)
            prev_ts = reading['timestamp']
            yield reading

    def close(self):
        self.conn.close()
