"""
Tests for Thermal Governor
============================
Covers: feature engineering, anomaly detection, state transitions, database.
"""

import unittest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from governor import (
    GovernorConfig, ZoneConfig, FeatureEngine,
    AnomalyDetector, GovernorState, SensorReader
)
from database import ThermalDatabase


class TestFeatureEngine(unittest.TestCase):
    """Test the time-series feature extraction pipeline."""

    def setUp(self):
        self.engine = FeatureEngine(window_size=60)

    def test_slope_rising(self):
        """Linearly rising temps should produce positive slope."""
        for i in range(20):
            self.engine.push('cpu', 50.0 + i * 0.5)
        slope = self.engine.get_slope('cpu')
        self.assertGreater(slope, 0.4)

    def test_slope_stable(self):
        """Stable temps should produce near-zero slope."""
        for _ in range(20):
            self.engine.push('cpu', 65.0 + np.random.uniform(-0.1, 0.1))
        slope = self.engine.get_slope('cpu')
        self.assertAlmostEqual(slope, 0.0, places=1)

    def test_slope_insufficient_data(self):
        """Should return 0 with < 10 data points."""
        for i in range(5):
            self.engine.push('cpu', 50.0 + i)
        self.assertEqual(self.engine.get_slope('cpu'), 0.0)

    def test_acceleration_detection(self):
        """Quadratic rise should produce positive acceleration."""
        for i in range(30):
            self.engine.push('gpu', 50.0 + (i ** 2) * 0.01)
        accel = self.engine.get_acceleration('gpu')
        self.assertGreater(accel, 0)

    def test_volatility_stable(self):
        """Low-noise signal should have low volatility."""
        for _ in range(20):
            self.engine.push('ssd', 55.0)
        vol = self.engine.get_volatility('ssd')
        self.assertAlmostEqual(vol, 0.0)

    def test_feature_vector_shape(self):
        """Feature vector should have correct dimensions."""
        config = GovernorConfig()
        readings = {'cpu': 70.0, 'gpu': 65.0, 'ssd': 50.0}
        for _ in range(15):
            vec = self.engine.get_feature_vector(readings)
        # 3 zones * 4 features + 3 cross-zone = 15
        self.assertEqual(vec.shape[1], 15)


class TestAnomalyDetector(unittest.TestCase):
    """Test the dual-model anomaly detection system."""

    def setUp(self):
        self.config = GovernorConfig()
        self.detector = AnomalyDetector(self.config)

    def test_statistical_normal(self):
        """Normal temps should produce low confidence."""
        for _ in range(20):
            score = self.detector.statistical_check('cpu', 60.0, 0.1)
        self.assertLess(score, 0.5)

    def test_statistical_anomaly(self):
        """High temps should trigger anomaly."""
        for _ in range(20):
            score = self.detector.statistical_check('cpu', 90.0, 1.0)
        self.assertGreater(score, 0.8)

    def test_debounce_prevents_flicker(self):
        """Single spike shouldn't trigger anomaly."""
        for _ in range(15):
            self.detector.statistical_check('cpu', 60.0, 0.1)
        # One spike
        score = self.detector.statistical_check('cpu', 95.0, 2.0)
        # Should still be low due to debounce
        self.assertLess(score, 0.8)

    def test_ml_not_ready_initially(self):
        """ML score should be 0 before training."""
        vec = np.random.randn(1, 15)
        score = self.detector.ml_check(vec)
        self.assertEqual(score, 0.0)


class TestSensorReader(unittest.TestCase):
    """Test the sensor abstraction layer."""

    def test_simulation_output(self):
        """Simulation should return all configured zones."""
        config = GovernorConfig(simulation_mode=True)
        sensor = SensorReader(config)
        readings = sensor.read_all_zones(False)
        self.assertEqual(set(readings.keys()), {'cpu', 'gpu', 'ssd'})

    def test_simulation_bounds(self):
        """Simulated temps should stay within physical bounds."""
        config = GovernorConfig(simulation_mode=True)
        sensor = SensorReader(config)
        sensor.set_sim_workload(1.0)
        for _ in range(200):
            readings = sensor.read_all_zones(False)
            for temp in readings.values():
                self.assertGreaterEqual(temp, 30.0)
                self.assertLessEqual(temp, 110.0)


class TestDatabase(unittest.TestCase):
    """Test SQLite logging and retrieval."""

    def setUp(self):
        self.db = ThermalDatabase(":memory:")

    def tearDown(self):
        self.db.close()

    def test_log_and_retrieve(self):
        self.db.log_reading('cpu', 72.5, 0.3, 0.4, 0.2, 0.35, 'MONITORING')
        readings = self.db.get_readings('cpu', last_n=1)
        self.assertEqual(len(readings), 1)
        self.assertAlmostEqual(readings[0]['temperature'], 72.5)

    def test_event_logging(self):
        self.db.log_event('THROTTLING', 'cpu=92.3')
        events = self.db.get_events(last_n=1)
        self.assertEqual(events[0]['event_type'], 'THROTTLING')


if __name__ == '__main__':
    unittest.main()
