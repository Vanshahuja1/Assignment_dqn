# modules/logger.py
import os
import json
import csv
import shutil
import datetime
import numpy as np
from pathlib import Path

class MetricsLogger:
    """
    Professional Metrics Logging Layer for Multi-Operator Edge UPF Scheduling Simulator.
    Supports per-slot and per-episode logging with CSV and JSON exports.
    """
    def __init__(self, log_dir="logs", run_name=None):
        if run_name is None:
            run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories
        self.snapshots_dir = self.log_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        
        # In-memory storage for aggregates
        self.slot_data = []
        self.episode_data = []
        
        # Files
        self.slot_csv = self.log_dir / "slot_metrics.csv"
        self.episode_csv = self.log_dir / "episode_metrics.csv"
        self.summary_json = self.log_dir / "summary.json"
        
        self.slot_header_written = False
        self.episode_header_written = False

    def save_config_snapshot(self, params_file_path):
        """Saves a snapshot of the current configuration for reproducibility."""
        if os.path.exists(params_file_path):
            shutil.copy(params_file_path, self.snapshots_dir / "params_snapshot.py")
        
        # Also save a JSON version of key params if needed
        # (Assuming params are imported and available)
        try:
            import params
            config_dict = {k: v for k, v in vars(params).items() if not k.startswith("__")}
            with open(self.snapshots_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=4, default=str)
        except Exception as e:
            print(f"Warning: Could not save config JSON: {e}")

    def log_slot(self, slot, episode, metrics):
        """Logs metrics for a single time slot."""
        row = {"episode": episode, "slot": slot}
        row.update(metrics)
        self.slot_data.append(row)
        
        # Append to CSV incrementally to save memory
        self._append_to_csv(self.slot_csv, row, is_slot=True)

    def log_episode(self, episode, metrics):
        """Logs metrics for a full episode."""
        row = {"episode": episode}
        row.update(metrics)
        self.episode_data.append(row)
        
        # Append to CSV
        self._append_to_csv(self.episode_csv, row, is_slot=False)
        
        # Update JSON summary
        self._update_json_summary()

    def _append_to_csv(self, file_path, data_dict, is_slot=True):
        file_exists = os.path.isfile(file_path)
        header_written = self.slot_header_written if is_slot else self.episode_header_written
        
        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data_dict.keys())
            if not file_exists or not header_written:
                writer.writeheader()
                if is_slot: self.slot_header_written = True
                else: self.episode_header_written = True
            writer.writerow(data_dict)

    def _update_json_summary(self):
        """Writes the latest aggregates to a JSON file."""
        summary = {
            "total_episodes": len(self.episode_data),
            "latest_episode": self.episode_data[-1] if self.episode_data else None,
            "averages": self._calculate_averages()
        }
        with open(self.summary_json, "w") as f:
            json.dump(summary, f, indent=4, default=str)

    def _calculate_averages(self):
        if not self.episode_data:
            return {}
        
        # Filter numeric fields for averaging
        averages = {}
        for key in self.episode_data[0].keys():
            if isinstance(self.episode_data[0][key], (int, float)):
                vals = [d[key] for d in self.episode_data if key in d]
                averages[f"avg_{key}"] = np.mean(vals)
                averages[f"peak_{key}"] = np.max(vals)
        return averages

    def finalize(self):
        """Final clean up and reporting."""
        print(f"\n✅ Logging complete. Results saved to: {self.log_dir}")
        print(f"   - CSV Logs: {self.slot_csv.name}, {self.episode_csv.name}")
        print(f"   - JSON Summary: {self.summary_json.name}")
        print(f"   - Config Snapshot: {self.snapshots_dir}")
