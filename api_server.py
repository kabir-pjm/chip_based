"""
Thermal Governor API Server
=============================
REST API + WebSocket for the live dashboard.
Runs the governor in a background thread and streams telemetry.
"""

import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from governor import ThermalGovernor, GovernorConfig


class GovernorAPI:
    """
    Lightweight API that wraps the governor and exposes telemetry.
    Writes snapshots to a JSON file that the dashboard polls.
    """

    def __init__(self, simulation: bool = True):
        self.config = GovernorConfig(simulation_mode=simulation)
        self.governor = ThermalGovernor(self.config)
        self.latest_snapshot = {}
        self.snapshot_path = Path("data/live_snapshot.json")
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    def _run_governor(self):
        """Background thread: runs governor loop and writes snapshots."""
        while self.governor.running:
            snapshot = self.governor.tick()

            # Add zone histories for charting
            snapshot['histories'] = {}
            for zone_id in self.config.zones:
                snapshot['histories'][zone_id] = self.governor.features.get_history(zone_id)

            self.latest_snapshot = snapshot

            # Write to file for dashboard polling
            with open(self.snapshot_path, 'w') as f:
                json.dump(snapshot, f)

            time.sleep(self.config.poll_interval)

    def start(self):
        """Start governor in background + serve dashboard."""
        thread = threading.Thread(target=self._run_governor, daemon=True)
        thread.start()

        # Serve dashboard files
        dashboard_dir = Path(__file__).parent.parent / "dashboard"
        import os
        os.chdir(dashboard_dir)

        server = HTTPServer(('localhost', 8080), SimpleHTTPRequestHandler)
        print("\n" + "=" * 60)
        print("  THERMAL GOVERNOR DASHBOARD")
        print("  Open: http://localhost:8080")
        print("=" * 60 + "\n")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.governor.running = False
            self.governor.actuator.resume_all()
            self.governor.db.close()
            print("\nShutdown complete.")


if __name__ == "__main__":
    GovernorAPI(simulation=True).start()
