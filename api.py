import os
import uvicorn
import json
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

from app.model_manager import ModelManager
from app.guardrails import check_guardrails, scrub_user_input
from app.rag import init_rag
from app.tools import run_tool, ALLOWED_TOOLS
from app.main import ask_model

app = FastAPI(title="Powerlifting Coach API")


class AskRequest(BaseModel):
    user_input: str = Field(..., example="Jakie jest optymalne tętno przy martwym ciągu?")
    k: int = Field(default=3, ge=1, le=10, description="Liczba dokumentów do pobrania z RAG")
    mode: Literal["precise", "creative"] = Field(default="precise", description="Tryb odpowiedzi")
    use_functions: bool = Field(default=True, description="Czy model ma używać narzędzi (RAG, kalkulator)?")

model_manager = None


@app.on_event("startup")
async def startup_event():
    global model_manager
    print(">>> Init API...")
    model_manager = ModelManager()
    pdf_path = os.getenv("KNOWLEDGE_PDF", "data/poliquin_picp_level_1.pdf")
    init_rag(pdf_path)
    print(">>> API ready.")


@app.post("/ask")
async def ask_endpoint(request: AskRequest):
    """Main endpoint to handle user queries."""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model nie został zainicjalizowany.")

    should_block, reason, flags = check_guardrails(request.user_input)

    if should_block:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Security Block",
                "message": reason,
                "flags": flags
            }
        )

    clean_input = scrub_user_input(request.user_input)

    temperature = 0.0 if request.mode == "precise" else 0.7

    if not request.use_functions:
        response = model_manager.chat(
            messages=[
                {"role": "system", "content": "You are a helpful Powerlifting Coach."},
                {"role": "user", "content": clean_input}
            ],
            temperature=temperature
        )
        return {
            "response": response["text"],
            "tool_used": None,
            "mode": request.mode
        }


    decision = ask_model(model_manager, clean_input)
    tool_name = decision.get("tool")

    if tool_name == "chat" or tool_name == "error":
        return {
            "response": decision.get("response", "Parsing error."),
            "tool_used": "chat"
        }

    if tool_name in ALLOWED_TOOLS:
        tool_args = decision.get("args", {})


        if tool_name == "kb_lookup":
            tool_args["top_k"] = request.k
            # print("f> [DEBUG] RAG top_k set to", request.k)

        tool_result = run_tool(tool_name, tool_args)

        if tool_result.get("status") == "error":
            return {
                "response": f"Tool error: {tool_result.get('message')}",
                "tool_used": tool_name,
                "details": tool_result
            }

        actual_result = tool_result.get("result", {})

        final_text = ""

        if tool_name == "kb_lookup" and actual_result.get("status") == "success":
            enhanced_prompt = actual_result.get("prompt")

            final_resp = model_manager.chat(
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=temperature,
                max_tokens=512
            )
            final_text = final_resp["text"]
        else:
            final_prompt = (
                f"User asked: {clean_input}\n"
                f"Tool Result: {json.dumps(actual_result)}\n"
                f"Explain this result to the user naturally."
            )
            final_resp = model_manager.chat(
                messages=[{"role": "user", "content": final_prompt}],
                temperature=temperature,
                max_tokens=512
            )
            final_text = final_resp["text"]

        return {
            "response": final_text,
            "tool_used": tool_name,
            "tool_params": tool_args,
            "mode": request.mode
        }

    return {"response": "Unknown tool.", "tool_used": "unknown"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)