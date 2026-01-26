import os
import time
from typing import List, Dict, Optional, Any

import torch
from dotenv import load_dotenv
from groq import Groq
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

class ModelManager:
    def __init__(self, mode: str = None):
        self.mode = mode or os.getenv("MODEL_MODE", "groq")
        self.groq_client = None
        self.groq_model_name = None
        self.local_model = None
        self.local_tokenizer = None
        self.local_device = None

        print(f" > [ModelManager] initialized in '{self.mode}' mode.")

        if self.mode == "groq":
            self._init_groq()
        elif self.mode == "local":
            self._init_local()
        else:
            raise ValueError(f"Unsupported MODEL_MODE: {self.mode}")

    def _init_groq(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        self.groq_model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
        self.groq_client = Groq(api_key=api_key)
        print(f" > [ModelManager] Initialized Groq client with model '{self.groq_model_name}'")


    def _init_local(self):
        self.local_model_name = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
        print(f" > [ModelManager] Loading local model '{self.local_model_name}'...")

        self.local_tokenizer = AutoTokenizer.from_pretrained(
            self.local_model_name,
            trust_remote_code=True
        )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        self.local_model = AutoModelForCausalLM.from_pretrained(
            self.local_model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map
        )

        self.local_device = "cuda" if torch.cuda.is_available() else "cpu"
        # Move model to device only if device_map is not used
        if device_map is None:
            self.local_model = self.local_model.to(self.local_device)

        print(f" > [ModelManager] Local model loaded on device '{self.local_device}'")
        print(f" > [ModelManager] Local model '{self.local_model_name}' is ready.")


    def _build_prompt_local(self, messages: List[Dict[str, str]]) -> str:
        """Builds the prompt text for local model from messages."""
        if hasattr(self.local_tokenizer, "apply_chat_template"):
            return self.local_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        return f"[SYSTEM]\n{system}\n[USER]\n{user}\n[ASSISTANT]\n"

    def _count_tokens_local(self, text: str) -> int:
        return len(self.local_tokenizer.encode(text))

    @torch.inference_mode()
    def _chat_local(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        json_mode: bool = False,
        json_schema: Optional[Dict] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        prompt_text = self._build_prompt_local(messages)
        inputs = self.local_tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        inputs = inputs.to(self.local_device)


        do_sample = temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "eos_token_id": self.local_tokenizer.eos_token_id,
            "pad_token_id": self.local_tokenizer.eos_token_id,
        }

        # Add sampling parameters if sampling is enabled
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k is not None:
                gen_kwargs["top_k"] = int(top_k)

        output_ids = self.local_model.generate(**inputs, **gen_kwargs)
        # Extract generated tokens (excluding prompt tokens)
        generated_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        output_text = self.local_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        dt = time.perf_counter() - t0

        prompt_tokens = sum(self._count_tokens_local(m["content"]) for m in messages)
        completion_tokens = self._count_tokens_local(output_text)

        return {
            "text": output_text,
            "latency_s": round(dt, 3),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    def _chat_groq(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.0,
            max_tokens: int = 256,
            top_p: float = 1.0,
            json_mode: bool = False
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            request_kwargs = {
                "model": self.groq_model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            if json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}

            response = self.groq_client.chat.completions.create(**request_kwargs)
            dt = time.perf_counter() - t0

            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            usage_dict = usage.model_dump() if usage else None

            return {
                "text": choice.message.content,
                "latency_s": round(dt, 3),
                "usage": usage_dict
            }

        except Exception as e:
            print(f"Groq LLM Error: {e}")
            return {
                "text": "I couldn't process that request.",
                "latency_s": round(time.perf_counter() - t0, 3),
                "usage": None
            }

    def chat(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.0,
            max_tokens: int = 256,
            top_p: float = 0.9,
            top_k: Optional[int] = None,
            json_mode: bool = False,
            json_schema: Optional[Dict] = None
    ) -> Dict[str, Any]:
        if self.mode == "groq":
            return self._chat_groq(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                json_mode=json_mode
            )
        elif self.mode == "local":
            return self._chat_local(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                json_mode=json_mode,
                json_schema=json_schema
            )
        else:
            raise ValueError(f"Unsupported MODEL_MODE: {self.mode}")

    def close(self):
        if self.mode == "groq":
            try:
                self.groq_client.close()
            except Exception as e:
                print(f" > [ERROR] closing Groq client: {e}")

        if self.local_model:
            del self.local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()