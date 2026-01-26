import csv
import os
from datetime import datetime
from collections import Counter
from typing import Dict, Any

LOG_FILE = "tool_logs.csv"
FIELDNAMES = ["timestamp", "tool_name", "status", "duration_s", "error_type", "tokens"]


def log_tool_execution(tool_name: str, status: str, duration: float, error_type: str = "", tokens: int = 0):
    file_exists = os.path.exists(LOG_FILE)

    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tool_name": tool_name,
                "status": status,
                "duration_s": round(duration, 3),
                "error_type": error_type,
                "tokens": tokens
            })
    except Exception as e:
        print(f"Logging failed: {e}")


def get_metrics_report() -> Dict[str, Any]:
    if not os.path.exists(LOG_FILE):
        return {"message": "No logs available."}

    total_calls = 0
    status_counts = Counter()
    tool_usage = Counter()
    error_types = Counter()
    total_duration = 0.0
    total_tokens = 0
    try:
        with open(LOG_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_calls += 1
                status = row["status"]
                tool = row["tool_name"]
                duration = float(row["duration_s"])
                err_type = row["error_type"]
                token = int(row["tokens"])
                total_tokens += token
                status_counts[status] += 1
                tool_usage[tool] += 1
                total_duration += duration
                if err_type:
                    error_types[err_type] += 1

    except Exception:
        return {"error": "Failed to parse logs."}

    if total_calls == 0:
        return {"message": "Log file is empty."}

    success_rate = (status_counts["SUCCESS"] / total_calls) * 100
    timeout_rate = (status_counts["TIMEOUT"] / total_calls) * 100
    error_rate = 100 - success_rate - timeout_rate

    return {
        "summary": {
            "total_calls": total_calls,
            "avg_latency_s": round(total_duration / total_calls, 3),
            "total_tokens": total_tokens
        },
        "rates": {
            "success": f"{success_rate:.1f}%",
            "timeout": f"{timeout_rate:.1f}%",
            "error": f"{error_rate:.1f}%"
        },
        "breakdown": dict(status_counts),
        "tools": dict(tool_usage.most_common()),
        "errors": dict(error_types)
    }