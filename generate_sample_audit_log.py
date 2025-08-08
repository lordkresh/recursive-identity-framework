# generate_sample_audit_log.py
# Run this once to create a realistic sample audit log for the visualizer.

import os
import json
from datetime import datetime, timedelta
import random

os.makedirs("demo", exist_ok=True)

base_time = datetime.now() - timedelta(days=14)
entries = []

# Simulate realistic RIF drift/compression patterns
drift_value = 0.05
ratio_value = 2.5

for i in range(15):
    timestamp = (base_time + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")

    # Drift tends to increase gradually with occasional resets
    drift_value += random.uniform(0.0, 0.05)
    if random.random() < 0.15:  # simulate identity refresh
        drift_value = random.uniform(0.0, 0.1)

    # Compression ratio might improve slightly with time
    ratio_value += random.uniform(-0.1, 0.15)
    ratio_value = max(1.5, min(ratio_value, 4.0))

    entries.append({
        "timestamp": timestamp,
        "drift": round(drift_value, 3),
        "compression_ratio": round(ratio_value, 2)
    })

with open("demo/audit_log.json", "w") as f:
    json.dump(entries, f, indent=2)

print("✅ Generated realistic sample audit_log.json in demo/")
