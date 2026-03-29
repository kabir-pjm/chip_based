"""
Predictive Thermal Governor with ML-Based Anomaly Detection
============================================================
A real-time anomaly detection and automated response system that monitors
hardware thermal zones, predicts dangerous trends using both statistical
and machine learning approaches, and takes corrective action.

Architecture:
    Sensor Layer → Feature Engineering → Anomaly Detection → Decision Engine → Actuator

Author: [Your Name]
License: MIT
"""

import os
import signal
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

import numpy as np
from sklearn.ensemble import IsolationForest

from database import ThermalDatabase

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class ZoneConfig:
    """Per-zone thermal thresholds and tuning parameters."""
    name: str
    safe_temp: float
    critical_temp: float
    predictive_slope: float = 0.5
    weight: float = 1.0  # importance weight for composite scoring


@dataclass
class GovernorConfig:
    """Global governor configuration."""
    window_size: int = 60
    poll_interval: float = 0.5
    debounce_threshold: float = 0.8
    ml_contamination: float = 0.05  # expected anomaly rate for Isolation Forest
    ml_retrain_interval: int = 200  # retrain every N ticks
    simulation_mode: bool = False
    zones: Dict[str, ZoneConfig] = field(default_factory=lambda: {
        'cpu': ZoneConfig(name='CPU', safe_temp=80.0, critical_temp=95.0, weight=1.0),
        'gpu': ZoneConfig(name='GPU', safe_temp=85.0, critical_temp=100.0, weight=0.8),
        'ssd': ZoneConfig(name='SSD', safe_temp=70.0, critical_temp=85.0, weight=0.6),
    })


class GovernorState(Enum):
    MONITORING = "MONITORING"
    WARNING = "WARNING"
    THROTTLING = "THROTTLING"
    CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────
# Sensor Abstraction Layer
# ─────────────────────────────────────────────

class SensorReader:
    """
    Abstraction over hardware sensors. Supports real hardware via psutil
    and a simulation mode for testing and demos.
    """

    def __init__(self, config: GovernorConfig):
        self.config = config
        self.sim_temps = {zone: 45.0 for zone in config.zones}
        self._sim_workload = 0.3

    def set_sim_workload(self, workload: float):
        self._sim_workload = max(0.0, min(1.0, workload))

    def read_all_zones(self, is_throttled: bool) -> Dict[str, float]:
        """Read temperatures from all configured zones."""
        if self.config.simulation_mode:
            return self._simulate(is_throttled)
        return self._read_hardware()

    def _simulate(self, is_throttled: bool) -> Dict[str, float]:
        """Generate realistic simulated temperatures."""
        results = {}
        for zone_id, zone_cfg in self.config.zones.items():
            t = self.sim_temps[zone_id]
            if is_throttled:
                t -= 1.8 + np.random.uniform(0, 0.5)
            else:
                heat = self._sim_workload * 1.8
                dissipation = 0.4
                noise = np.random.uniform(-0.3, 0.3)
                t += heat - dissipation + noise

            # Each zone has different thermal characteristics
            if zone_id == 'gpu':
                t += np.random.uniform(-0.2, 0.4)  # GPUs are spikier
            elif zone_id == 'ssd':
                t *= 0.85  # SSDs run cooler

            t = np.clip(t, 35.0, zone_cfg.critical_temp + 5)
            self.sim_temps[zone_id] = t
            results[zone_id] = round(t, 2)
        return results

    def _read_hardware(self) -> Dict[str, float]:
        """Read from actual hardware sensors via psutil."""
        import psutil
        results = {}
        try:
            sensors = psutil.sensors_temperatures()
            zone_map = {
                'coretemp': 'cpu', 'k10temp': 'cpu',
                'amdgpu': 'gpu', 'nouveau': 'gpu',
                'nvme': 'ssd', 'drivetemp': 'ssd',
            }
            for sensor_name, entries in sensors.items():
                zone_id = zone_map.get(sensor_name)
                if zone_id and zone_id in self.config.zones:
                    temps = [e.current for e in entries if e.current > 0]
                    if temps:
                        results[zone_id] = max(temps)
        except Exception as e:
            logging.error(f"Sensor read error: {e}")

        # Fill missing zones with 0
        for zone_id in self.config.zones:
            if zone_id not in results:
                results[zone_id] = 0.0
        return results


# ─────────────────────────────────────────────
# Feature Engineering Pipeline
# ─────────────────────────────────────────────

class FeatureEngine:
    """
    Extracts time-series features from raw temperature readings.
    Features: slope, acceleration, volatility, zone spread, rolling stats.
    """

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.histories: Dict[str, deque] = {}

    def _ensure_zone(self, zone_id: str):
        if zone_id not in self.histories:
            self.histories[zone_id] = deque(maxlen=self.window_size)

    def push(self, zone_id: str, temp: float):
        self._ensure_zone(zone_id)
        self.histories[zone_id].append(temp)

    def get_slope(self, zone_id: str) -> float:
        """Linear regression slope — rate of temperature change."""
        self._ensure_zone(zone_id)
        h = self.histories[zone_id]
        if len(h) < 10:
            return 0.0
        y = np.array(list(h))
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]

    def get_acceleration(self, zone_id: str) -> float:
        """Second derivative — is the heating accelerating?"""
        self._ensure_zone(zone_id)
        h = self.histories[zone_id]
        if len(h) < 15:
            return 0.0
        y = np.array(list(h))
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 2)
        return coeffs[0]

    def get_volatility(self, zone_id: str, lookback: int = 10) -> float:
        """Standard deviation of recent readings — sensor noise indicator."""
        self._ensure_zone(zone_id)
        h = list(self.histories[zone_id])
        if len(h) < lookback:
            return 0.0
        return float(np.std(h[-lookback:]))

    def get_feature_vector(self, readings: Dict[str, float]) -> np.ndarray:
        """
        Build a feature vector from current readings for ML model.
        Features per zone: [temp, slope, acceleration, volatility]
        Plus cross-zone: [max_temp, zone_spread, max_slope]
        """
        features = []
        all_temps = []
        all_slopes = []

        for zone_id, temp in readings.items():
            self.push(zone_id, temp)
            slope = self.get_slope(zone_id)
            accel = self.get_acceleration(zone_id)
            vol = self.get_volatility(zone_id)
            features.extend([temp, slope, accel, vol])
            all_temps.append(temp)
            all_slopes.append(slope)

        # Cross-zone features
        features.append(max(all_temps) if all_temps else 0)
        features.append(max(all_temps) - min(all_temps) if len(all_temps) > 1 else 0)
        features.append(max(all_slopes) if all_slopes else 0)

        return np.array(features).reshape(1, -1)

    def get_history(self, zone_id: str) -> List[float]:
        self._ensure_zone(zone_id)
        return list(self.histories[zone_id])


# ─────────────────────────────────────────────
# Anomaly Detection (Statistical + ML)
# ─────────────────────────────────────────────

class AnomalyDetector:
    """
    Dual-model anomaly detection:
    1. Statistical: slope + threshold with debounce filtering
    2. Machine Learning: Isolation Forest on multi-dimensional features

    The final decision fuses both signals for robust detection.
    """

    def __init__(self, config: GovernorConfig):
        self.config = config
        self.alert_buffers: Dict[str, deque] = {}
        self.ml_model: Optional[IsolationForest] = None
        self.training_buffer: List[np.ndarray] = []
        self.ml_ready = False
        self.tick_count = 0

    def _ensure_buffer(self, zone_id: str):
        if zone_id not in self.alert_buffers:
            self.alert_buffers[zone_id] = deque(maxlen=self.config.window_size // 2)

    def statistical_check(self, zone_id: str, temp: float, slope: float) -> float:
        """
        Returns a confidence score (0.0 - 1.0) based on threshold + debounce.
        """
        self._ensure_buffer(zone_id)
        zone_cfg = self.config.zones.get(zone_id)
        if not zone_cfg:
            return 0.0

        is_hot = temp > zone_cfg.safe_temp
        is_rising = slope > zone_cfg.predictive_slope
        danger = 1 if (is_hot or is_rising) else 0
        self.alert_buffers[zone_id].append(danger)

        buf = self.alert_buffers[zone_id]
        if len(buf) < 5:
            return 0.0
        return sum(buf) / len(buf)

    def ml_check(self, feature_vector: np.ndarray) -> float:
        """
        Isolation Forest anomaly score. Returns 0.0 (normal) to 1.0 (anomalous).
        """
        if not self.ml_ready or self.ml_model is None:
            return 0.0

        # decision_function: negative = anomaly, positive = normal
        score = self.ml_model.decision_function(feature_vector)[0]
        # Normalize to 0-1 where 1 = most anomalous
        return float(np.clip(-score, 0, 1))

    def train_ml_model(self):
        """Retrain Isolation Forest on accumulated data."""
        if len(self.training_buffer) < 50:
            return
        X = np.vstack(self.training_buffer)
        self.ml_model = IsolationForest(
            contamination=self.config.ml_contamination,
            n_estimators=100,
            random_state=42
        )
        self.ml_model.fit(X)
        self.ml_ready = True
        logging.info(f"ML model retrained on {len(X)} samples")

    def detect(self, zone_id: str, temp: float, slope: float,
               feature_vector: np.ndarray) -> Dict:
        """
        Fused anomaly detection combining statistical and ML signals.
        Returns detection result with component scores.
        """
        self.tick_count += 1

        # Collect training data
        self.training_buffer.append(feature_vector.flatten())
        if len(self.training_buffer) > 1000:
            self.training_buffer = self.training_buffer[-1000:]

        # Periodically retrain
        if self.tick_count % self.config.ml_retrain_interval == 0:
            self.train_ml_model()

        stat_score = self.statistical_check(zone_id, temp, slope)
        ml_score = self.ml_check(feature_vector)

        # Fusion: weighted combination. ML gets more weight once trained.
        if self.ml_ready:
            fused = 0.5 * stat_score + 0.5 * ml_score
        else:
            fused = stat_score

        return {
            'statistical_score': round(stat_score, 3),
            'ml_score': round(ml_score, 3),
            'fused_score': round(fused, 3),
            'ml_active': self.ml_ready,
            'is_anomaly': fused >= self.config.debounce_threshold,
        }


# ─────────────────────────────────────────────
# Process Actuator
# ─────────────────────────────────────────────

class ProcessActuator:
    """
    Manages process suspension and resumption via POSIX signals.
    SIGSTOP suspends without data loss. SIGCONT resumes cleanly.
    """

    def __init__(self):
        self.throttled_pids: set = set()

    def find_targets(self, target_name: str = "stress_test") -> List[int]:
        """Find processes matching the target pattern."""
        import psutil
        pids = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                info = proc.info
                name_match = target_name in (info['name'] or '')
                cmd_match = info['cmdline'] and any(
                    target_name in cmd for cmd in info['cmdline']
                )
                if name_match or cmd_match:
                    pids.append(info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return pids

    def suspend(self, pids: List[int]):
        """Suspend processes with SIGSTOP."""
        for pid in pids:
            if pid not in self.throttled_pids:
                try:
                    os.kill(pid, signal.SIGSTOP)
                    self.throttled_pids.add(pid)
                    logging.warning(f"Process {pid} suspended (SIGSTOP)")
                except OSError as e:
                    logging.error(f"Failed to suspend {pid}: {e}")

    def resume_all(self):
        """Resume all suspended processes with SIGCONT."""
        if not self.throttled_pids:
            return
        logging.info("Resuming all suspended processes")
        for pid in list(self.throttled_pids):
            try:
                os.kill(pid, signal.SIGCONT)
                logging.info(f"Process {pid} resumed (SIGCONT)")
            except OSError:
                pass
            finally:
                self.throttled_pids.discard(pid)

    @property
    def is_throttling(self) -> bool:
        return len(self.throttled_pids) > 0


# ─────────────────────────────────────────────
# Main Governor
# ─────────────────────────────────────────────

class ThermalGovernor:
    """
    Orchestrates the full pipeline:
    Sensors → Features → Anomaly Detection → Decision → Actuation → Logging
    """

    def __init__(self, config: Optional[GovernorConfig] = None):
        self.config = config or GovernorConfig()
        self.sensor = SensorReader(self.config)
        self.features = FeatureEngine(self.config.window_size)
        self.detector = AnomalyDetector(self.config)
        self.actuator = ProcessActuator()
        self.db = ThermalDatabase()
        self.state = GovernorState.MONITORING
        self.running = True

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [GOVERNOR] %(levelname)s %(message)s'
        )
        logging.info(f"Governor initialized | Zones: {list(self.config.zones.keys())}")
        logging.info(f"ML contamination: {self.config.ml_contamination}")
        if self.config.simulation_mode:
            logging.warning("SIMULATION MODE ACTIVE")

    def _determine_state(self, detections: Dict[str, Dict]) -> GovernorState:
        """Determine overall governor state from per-zone detections."""
        any_anomaly = any(d['is_anomaly'] for d in detections.values())
        max_fused = max((d['fused_score'] for d in detections.values()), default=0)

        if any_anomaly and self.actuator.is_throttling:
            return GovernorState.THROTTLING
        elif any_anomaly:
            return GovernorState.WARNING
        elif max_fused > 0.5:
            return GovernorState.WARNING
        return GovernorState.MONITORING

    def tick(self) -> Dict:
        """
        Execute one monitoring cycle. Returns full telemetry snapshot.
        """
        readings = self.sensor.read_all_zones(self.actuator.is_throttling)
        feature_vector = self.features.get_feature_vector(readings)

        detections = {}
        for zone_id, temp in readings.items():
            slope = self.features.get_slope(zone_id)
            det = self.detector.detect(zone_id, temp, slope, feature_vector)
            detections[zone_id] = {**det, 'temp': temp, 'slope': round(slope, 4)}

        self.state = self._determine_state(detections)

        # Act on anomalies
        if any(d['is_anomaly'] for d in detections.values()):
            targets = self.actuator.find_targets()
            if targets:
                self.actuator.suspend(targets)
        else:
            self.actuator.resume_all()

        # Log to database
        for zone_id, det in detections.items():
            self.db.log_reading(
                zone=zone_id,
                temperature=det['temp'],
                slope=det['slope'],
                stat_score=det['statistical_score'],
                ml_score=det['ml_score'],
                fused_score=det['fused_score'],
                state=self.state.value
            )

        if self.state != GovernorState.MONITORING:
            self.db.log_event(
                event_type=self.state.value,
                details=str({z: round(d['temp'], 1) for z, d in detections.items()})
            )

        return {
            'state': self.state.value,
            'zones': detections,
            'ml_active': self.detector.ml_ready,
            'throttled_pids': list(self.actuator.throttled_pids),
        }

    def run(self):
        """Main loop."""
        logging.info("Governor daemon started")
        try:
            while self.running:
                snapshot = self.tick()
                if snapshot['state'] != 'MONITORING':
                    logging.warning(f"State: {snapshot['state']} | "
                                    f"Zones: {snapshot['zones']}")
                time.sleep(self.config.poll_interval)
        except KeyboardInterrupt:
            logging.info("Shutting down — resuming all processes")
            self.actuator.resume_all()
            self.db.close()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predictive Thermal Governor")
    parser.add_argument('--sim', action='store_true', help='Run in simulation mode')
    parser.add_argument('--interval', type=float, default=0.5, help='Poll interval (seconds)')
    parser.add_argument('--safe-temp', type=float, default=80.0, help='Safe temperature threshold')
    args = parser.parse_args()

    config = GovernorConfig(
        simulation_mode=args.sim,
        poll_interval=args.interval,
    )
    if args.safe_temp != 80.0:
        for zone in config.zones.values():
            zone.safe_temp = args.safe_temp

    if not args.sim and os.geteuid() != 0:
        print("Error: Hardware mode requires root (sudo). Use --sim for simulation.")
        exit(1)

    ThermalGovernor(config).run()
