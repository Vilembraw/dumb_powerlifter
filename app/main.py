import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

from app.rag import init_rag
from app.tools import run_tool, ALLOWED_TOOLS
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file!")

client = Groq(api_key=API_KEY)

SYSTEM_PROMPT = """
You are an expert AI Powerlifting Coach.
Your goal is to assist users with training math, rules, and logging.

AVAILABLE TOOLS:
1. 'calculate_1rm': Calculate One Rep Max or weight for reps. 
   Args: {"weight": float, "reps": int, "mode": "calculate_max" | "calculate_reps_weight"}
2. 'log_workout': Log a workout to CSV.
   Args: {"exercises": [{"exercise": str, "weight": float, "reps": int}]}
3. 'kb_lookup': Search for rules, definitions, technique tips, or RPE explanation.
   Args: {"query": str}

INSTRUCTIONS:
- If the user asks for a calculation or logging, RETURN A JSON OBJECT with the tool name and arguments.
- If no tool is needed (general chat), return a JSON with "tool": "chat" and "response": "your message".
- If user asks about RULES, TECHNIQUE, DEFINITIONS (e.g., "squat depth", "what is RPE") -> use 'kb_lookup'.
- STRICTLY output JSON. No markdown code blocks.

FORMAT:
{"tool": "tool_name", "args": { ... }}
OR
{"tool": "chat", "response": "..."}
"""

def ask_groq_router(user_input: str):
    """
    Sends the user input to groq and forces a JSON response.
    This implements the 'Router'
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM Error: {e}")
        return {"tool": "error", "response": "I couldn't process that request."}


def main():
    pdf_path = os.getenv("KNOWLEDGE_PDF", "data/poliquin_picp_level_1.pdf")
    init_rag(pdf_path)

    print(f"--- AI Powerlifting Coach ({MODEL_NAME}) ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # A. ROUTING (LLM Decision)
        decision = ask_groq_router(user_input)

        tool_name = decision.get("tool")

        # B. DISPATCHING (Execution)
        if tool_name in ALLOWED_TOOLS:
            print(f" > [DEBUG] Calling Tool: {tool_name} with {decision.get('args')}")

            tool_result = run_tool(tool_name, decision.get("args"), timeout_s=5.0)

            if tool_result.get("status") == "error":
                error_type = tool_result.get("error_type", "unknown")
                message = tool_result.get("message", "Unknown error")

                if error_type == "timeout":
                    print(f"\nCoach: Sorry, the tool '{tool_name}' took too long to respond.\n")
                elif error_type == "security_blocked":
                    print(f"\nCoach: {message}\n")
                elif error_type == "validation_error":
                    print(f"\nCoach: Invalid input:  {message}\n")
                    print(f"Details: {tool_result.get('details', 'N/A')}\n")
                else:
                    print(f"\nCoach: Something went wrong: {message}\n")
                continue

            actual_result = tool_result.get("result", {})

            if tool_name == "kb_lookup" and actual_result.get("status") == "success":
                # Use the enhanced RAG prompt directly
                enhanced_prompt = actual_result.get("prompt")

                final_resp = client.chat.completions.create(  # Changed here
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    temperature=0.3,
                    max_tokens=1024
                )

                print(f"\nCoach: {final_resp.choices[0].message.content}\n")

                print(f"Sources:  {len(actual_result['hits'])} passages from knowledge base")
                for i, hit in enumerate(actual_result['hits'], 1):
                    print(f"[{i}] Page {hit['page']} (Score: {hit['score']:.2f})")
                print()
            else:
                #C. Standard synthesis for other tools
                final_prompt = f"User asked:  {user_input}\nTool Result: {json.dumps(actual_result)}\nExplain this result to the user naturally."

                final_resp = client.chat.completions.create(  # Changed here
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.3
                )
                print(f"\nCoach: {final_resp.choices[0].message.content}\n")


            # D. OBSERVABILITY
            # TODO: Logs to csv or monitoring system can be added here


        elif tool_name == "chat":
            print(f"Coach: {decision.get('response')}\n")

        else:
            print("Coach: I am confused. Please try again.\n")


if __name__ == "__main__":
    main()