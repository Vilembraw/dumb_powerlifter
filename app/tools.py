import os
import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from typing import Literal, Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError

from app.observability import log_tool_execution
from app.rag import search_knowledge


class Calc1RMArgs(BaseModel):
    weight: float = Field(...,gt=0, le=500, description="Weight in kg or lbs")
    reps: int = Field(..., gt=0, le=100, description="Number of repetitions (must be > 0)")
    mode: Literal["calculate_max", "calculate_reps_weight"] = Field(
        ..., description="'calculate_max' computes 1RM. 'calculate_reps_weight' computes weight for reps."
    )

class LogWorkoutArgs(BaseModel):
    exercises: List[Dict[str, Any]] = Field(..., description="List of exercises. Each must contain: exercise, weight, reps.")

class KBLookupArgs(BaseModel):
    query: str = Field(..., description="Topic or question to search in Knowledge Base (e.g. 'squat rules', 'what is RPE')")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top results to retrieve from KB")

TOOL_TIMEOUTS = {
    "calculate_1rm": 2.0,
    "log_workout": 3.0,
    "kb_lookup": 15.0,
}
DEFAULT_TIMEOUT = 5.0

def _kb_lookup(args: KBLookupArgs, cancel_event: Optional[threading. Event] = None) -> Dict[str, Any]:
    """Retrieves information from the Knowledge Base."""

    # Check for cancellation before search
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Operation cancelled")

    # Search the knowledge base
    hits = search_knowledge(args.query, top_k=args.top_k, use_hybrid=True)

    # Check for cancellation after search
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Operation cancelled")

    # Check if any relevant hits were found
    if hits:
        print(f"Top hit score: {hits[0].get('score', 'N/A')}")
        print(f"Top hit preview: {hits[0].get('text', '')[:100]}...")
    else:
        print("No hits returned from search_knowledge()")


    RELEVANCE_THRESHOLD = 0.67

    if not hits or hits[0].get("score", 0) < RELEVANCE_THRESHOLD:
        return {
            "status": "success",
            "prompt": (
                f"The user asked: '{args.query}'.\n\n"
                f"I searched the knowledge base but found no highly relevant information "
                f"(best match score: {hits[0].get('score', 0):.2f} < {RELEVANCE_THRESHOLD}).\n\n"
                f"INSTRUCTION: Answer this question using your **general expert knowledge** as an experienced Powerlifting Coach.\n\n"
                f"CRITICAL RULES:\n"
                f"1. START your response with: \"Note: This is based on general coaching knowledge, not the provided documents.\"\n"
                f"2. Be helpful, detailed, and accurate\n"
                f"3. Use proper powerlifting terminology\n"
                f"4. Structure your answer with bullet points or numbered steps if explaining technique\n"
                f"5. Cite general principles (e.g., 'According to standard powerlifting coaching...')\n\n"
                f"USER QUESTION: {args.query}\n\n"
                f"ANSWER:"
            ),
            "hits": [],
            "query": args.query
        }

    # Format context from hits
    context_parts = []
    for i, hit in enumerate(hits, 1):
        # Check for cancellation during context assembly
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation cancelled")

        method = {
            "dense": "Semantic",
            "bm25": "Keyword",
            "both": "Hybrid",
            "unknown": "Unknown"
        }.get(hit.get("retrieved_by", "unknown"), "Unknown")

        context_parts.append(
            f"[{i}] ({method} Search, Page {hit['page']}, Score: {hit['score']:.2f})\n{hit['text']}"
        )

    context = "\n\n".join(context_parts)

    rag_prompt = f"""You are an expert AI Powerlifting Coach with deep knowledge of training principles, technique, and competition rules. 

    USER QUESTION: {args.query}

    RETRIEVED KNOWLEDGE BASE CONTEXT:
    {context}

    INSTRUCTIONS:
    1. Carefully analyze the retrieved context above
    2. Answer the user's question directly and comprehensively using ONLY information from the context
    3. Use specific details, numbers, and rules from the context
    4. Cite source page numbers when providing specific facts (e.g., "According to page 5...")
    5. If the context doesn't fully answer the question, acknowledge: "The provided materials don't cover [specific aspect]"
    6. Use clear, actionable language appropriate for a powerlifting athlete
    7. Structure your response with bullet points or numbered lists when appropriate
    8. Be concise - avoid repeating the same information
    
    CRITICAL GROUNDING RULES:
    - Stay STRICTLY within the knowledge provided in the context
    - Do NOT add information from general knowledge
    - If information is partial or unclear, say so explicitly
    - Use technical terminology accurately as presented in the sources
    
    ANSWER:"""

    return {
        "status": "success",
        "context": context,
        "hits": hits,
        "message": f"Found {len(hits)} relevant passages from the knowledge base.",
        "prompt": rag_prompt,
        "query": args.query
    }


def _calculate_1rm(args: Calc1RMArgs, cancel_event: Optional[threading. Event] = None) -> Dict[str, Any]:
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


def _log_workout(args: LogWorkoutArgs, cancel_event: Optional[threading. Event] = None) -> Dict[str, Any]:
    """Logs the workout to CSV."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, "user_progress.csv")
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

                if cancel_event and cancel_event.is_set():
                    os.remove(filename)
                    raise RuntimeError("Operation cancelled")

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
        if os.path.exists(filename):
            os.remove(filename)
        return {"status": "error", "msg": str(e)}

    return {
        "status": "success",
        "message": f"Saved {len(saved_entries)} sets: {', '.join(saved_entries)}."
    }


ALLOWED_TOOLS = {
    "calculate_1rm": (Calc1RMArgs, _calculate_1rm),
    "log_workout": (LogWorkoutArgs, _log_workout),
    "kb_lookup": (KBLookupArgs, _kb_lookup),
}


_cancellation_tokens: Dict[int, threading.Event] = {}

def _run_tool_sync(tool_name: str, validated_args, cancel_event: threading.Event) -> Dict[str, Any]:
    """Synchronous tool runner with cancellation support."""
    Schema, Function = ALLOWED_TOOLS[tool_name]
    # Setup cancellation event
    if hasattr(Function, '__wrapped__'):
        return Function(validated_args, cancel_event=cancel_event)
    else:
        return Function(validated_args)


def run_tool(tool_name: str, tool_args: Dict[str, Any], timeout_s: float = 5.0) -> Dict[str, Any]:
    """Runs a tool with validation, timeout, and cancellation support."""

    start_time = time.perf_counter()
    status_log = "ERROR"
    error_type_log = "unknown"

    if tool_args is None:
        tool_args = {}

    if tool_name not in ALLOWED_TOOLS:
        duration = time.perf_counter() - start_time
        log_tool_execution(tool_name, "BLOCKED", duration, "security_blocked")
        return {
            "status": "error",
            "error_type": "security_blocked",
            "message": f"Tool '{tool_name}' not allowed."
        }

    if timeout_s is None:
        timeout_s = TOOL_TIMEOUTS.get(tool_name, DEFAULT_TIMEOUT)
    Schema, Function = ALLOWED_TOOLS[tool_name]

    # Validation
    try:
        validated_args = Schema(**tool_args)
    except ValidationError as e:
        # Handling validation errors cleanly
        duration = time.perf_counter() - start_time
        log_tool_execution(tool_name, "ERROR", duration, "validation_error")
        return {
            "status": "error",
            "error_type": "validation_error",
            "message": "Invalid arguments",
            "details": e.errors()
        }

    cancel_event = threading.Event()
    thread_id = threading.get_ident()
    _cancellation_tokens[thread_id] = cancel_event
    effective_timeout = timeout_s or TOOL_TIMEOUTS.get(tool_name, DEFAULT_TIMEOUT)
    # Execute with timeout
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_tool_sync, tool_name, validated_args, cancel_event)
                try:
                    result = future.result(timeout=effective_timeout)
                    status_log = "SUCCESS"
                    error_type_log = ""
                    return {"status": "success", "result": result}

                # Handle timeout
                except FuturesTimeout:
                    cancel_event.set()
                    status_log = "TIMEOUT"
                    error_type_log = "timeout"

                    return {
                        "status": "error",
                        "error_type": "timeout",
                        "message": f"Tool '{tool_name}' exceeded {effective_timeout}s timeout."
                    }

                except Exception as e:
                    cancel_event.set()
                    status_log = "ERROR"
                    error_type_log = "tool_error"
                    return {
                        "status": "error",
                        "error_type": "tool_error",
                        "message": f"Tool '{tool_name}' failed",
                        "details": str(e)
                    }
    finally:
        _cancellation_tokens.pop(thread_id, None)
        duration = time.perf_counter() - start_time
        log_tool_execution(tool_name, status_log, duration, error_type_log)