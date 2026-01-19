import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from tools import run_tool, ALLOWED_TOOLS
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file!")

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = """
You are an expert AI Powerlifting Coach.
Your goal is to assist users with training math, rules, and logging.

AVAILABLE TOOLS:
1. 'calculate_1rm': Calculate One Rep Max or weight for reps. 
   Args: {"weight": float, "reps": int, "mode": "calculate_max" | "calculate_reps_weight"}
2. 'log_workout': Log a workout to CSV.
   Args: {"exercises": [{"exercise": str, "weight": float, "reps": int}]}

INSTRUCTIONS:
- If the user asks for a calculation or logging, RETURN A JSON OBJECT with the tool name and arguments.
- If no tool is needed (general chat), return a JSON with "tool": "chat" and "response": "your message".
- STRICTLY output JSON. No markdown code blocks.

FORMAT:
{"tool": "tool_name", "args": { ... }}
OR
{"tool": "chat", "response": "..."}
"""

def ask_gemini_router(user_input: str):
    """
    Sends the user input to Gemini and forces a JSON response.
    This implements the 'Router'
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0, # Deterministic for tools
                response_mime_type="application/json" # JSON Enforcement
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"LLM Error: {e}")
        return {"tool": "error", "response": "I couldn't process that request."}


def main():
    print(f"--- AI Powerlifting Coach ({MODEL_NAME}) ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # A. ROUTING (LLM Decision)
        decision = ask_gemini_router(user_input)

        tool_name = decision.get("tool")

        # B. DISPATCHING (Execution)
        if tool_name in ALLOWED_TOOLS:
            print(f" > [DEBUG] Calling Tool: {tool_name} with {decision.get('args')}")

            tool_result = run_tool(tool_name, decision.get("args"))

            # C. FINAL RESPONSE (Synthesis)
            # We feed the tool result back to the LLM to generate a natural answer
            final_prompt = f"User asked: {user_input}\nTool Result: {json.dumps(tool_result)}\nExplain this result to the user naturally."

            final_resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=final_prompt,
                config=types.GenerateContentConfig(temperature=0.3)
            )
            print(f"Coach: {final_resp.text}\n")

            # D. OBSERVABILITY
            # TODO: Logs to csv or monitoring system can be added here


        elif tool_name == "chat":
            print(f"Coach: {decision.get('response')}\n")

        else:
            print("Coach: I am confused. Please try again.\n")


if __name__ == "__main__":
    main()