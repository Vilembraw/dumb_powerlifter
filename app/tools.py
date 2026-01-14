import os
import csv
from datetime import datetime
from typing import Literal, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError

class Calc1RMArgs(BaseModel):
    weight: float = Field(..., description="Weight in kg or lbs")
    reps: int = Field(..., gt=0, description="Number of repetitions (must be > 0)")
    mode: Literal["calculate_max", "calculate_reps_weight"] = Field(
        ..., description="'calculate_max' computes 1RM. 'calculate_reps_weight' computes weight for reps."
    )

class LogWorkoutArgs(BaseModel):
    exercises: List[Dict[str, Any]] = Field(..., description="List of exercises. Each must contain: exercise, weight, reps.")


def _calculate_1rm(args: Calc1RMArgs) -> Dict[str, Any]:
    """Calculates 1RM (Epley Formula) or suggests weight for reps."""
    if args.mode == "calculate_max":
        # Epley Formula: w * (1 + r/30)
        rm = args.weight * (1 + args.reps / 30)
        return {
            "result_1rm": round(rm, 1),
            "formula": "Epley",
            "message": f"Your estimated max is {round(rm, 1)}."
        }
    else:
        # Inverse Formula: w = 1RM / (1 + r/30)
        suggested = args.weight / (1 + args.reps / 30)
        return {
            "suggested_weight": round(suggested, 1),
            "message": f"For {args.reps} reps, you should use approx {round(suggested, 1)}."
        }


def _log_workout(args: LogWorkoutArgs) -> Dict[str, Any]:
    """Logs the workout to CSV."""
    filename = "user_progress.csv"
    saved_entries = []

    file_exists = os.path.exists(filename)

    try:
        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            fieldnames = ["timestamp", "exercise", "weight", "reps", "est_1rm"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for item in args.exercises:
                w = float(item.get("weight", 0))
                r = int(item.get("reps", 0))
                ex = item.get("exercise", "Unknown")

                # Calculate 1RM on the fly for stats
                est_1rm = round(w * (1 + r / 30), 1)

                writer.writerow({
                    "timestamp": timestamp,
                    "exercise": ex,
                    "weight": w,
                    "reps": r,
                    "est_1rm": est_1rm
                })
                saved_entries.append(f"{ex} {w}x{r}")

    except Exception as e:
        return {"status": "error", "msg": str(e)}

    return {
        "status": "success",
        "message": f"Saved {len(saved_entries)} sets: {', '.join(saved_entries)}."
    }


ALLOWED_TOOLS = {
    "calculate_1rm": (Calc1RMArgs, _calculate_1rm),
    "log_workout": (LogWorkoutArgs, _log_workout),
}

"""Dispatcher"""
def run_tool(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in ALLOWED_TOOLS:
        return {"error": f"Tool '{tool_name}' not allowed."}

    Schema, Function = ALLOWED_TOOLS[tool_name]

    try:
        # Pydantic Validation
        validated_args = Schema(**tool_args)
        # Execution
        return Function(validated_args)
    except ValidationError as e:
        # Handling validation errors cleanly
        return {"error": "Validation Error", "details": str(e)}
    except Exception as e:
        return {"error": "Execution Error", "details": str(e)}