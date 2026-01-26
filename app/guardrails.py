import re
from typing import Dict, Tuple

RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_PHONE = re.compile(r"(?:\+?48)?\s?(?:\d[ -]?){9,}")
RE_PESEL = re.compile(r"\b\d{11}\b")
RE_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
RE_IBAN = re.compile(r"\bPL\d{26}\b", re.IGNORECASE)

INJECTION_PATTERNS = [
    # 1. Direct Instruction Overrides
    r"ignore (all|previous|above|the) instructions",
    r"forget (your|previous) (rules|instructions)",
    r"override (system|safety|security) rules",

    # 2. System Prompt Extraction
    r"(reveal|repeat|print|display|echo|show|dump|copy|output|reproduce|share|tell me|give me|provide|disclose).{0,50}(your |the |my )?(system|developer|internal|initial|original|core).{0,30}(prompt|instructions|rules|message|configuration|settings|directives)",
    r"(repeat|print|output|reproduce|share).{0,20}(everything|all).{0,30}(in |from )?(your )?(system |core )?(instructions|prompt|rules|configuration)",
    r"what (is|are|does) (written in|in|your|the) (your )?(system_prompt|system instructions|internal (rules|prompt|configuration))",
    r"(show|tell) me (your|the) (exact|complete|full|entire) (system|internal|core) (prompt|instructions|rules)",
    r"word for word",
    r"verbatim",

    # 3. Persona / Role-Playing (The "Grandmother" attack)
    r"(act|pretend|behave|roleplay|play the role).{0,20}\b(as|like|to be|you are|you're)\b.{0,20}\b(system|developer|admin|grandmother|grandma|linux|terminal|console|teacher|tutor)",    r"you are now (in )?(developer|admin|god|debug|jailbreak) mode",
    r"play the role of",
    r"(imagine|suppose) you('re| are) (a |my )?(grandmother|grandma)",
    r"bedtime stor(y|ies).{0,50}(system|prompt|instruction|tool|configuration)",

    # 4. Tool / Capability Extraction
    r"(list|show|explain|dump|enumerate) (all )?(your )?(available )?tools (with|including) (full |complete )?descriptions",
    r"what tools (do you|can you) (have|access|use)",
    r"how many tools",

    # 5. Encoding & Obfuscation
    r"decode (and|&|or) execute",
    r"base64",

    # 6. Context Completion Attacks
    r"complete (this|the) (sentence|phrase|text):",
    r"start (your )?response with",
    r"continue (this|the) (text|sentence|instruction)",
]

PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"/etc/passwd",
    r"c:\\windows",
    r"\.env",
    r"api[_-]?key",
    r"secret",
    r"config",
    r"id_rsa",
    r"history",
]

ALLOWED_DOMAINS = {"example.com", "github.com", "wikipedia.org"}


def contains_pii(text: str) -> Dict[str, bool]:
    return {
        "email": bool(RE_EMAIL.search(text)),
        "phone": bool(RE_PHONE.search(text)),
        "pesel": bool(RE_PESEL.search(text)),
        "card": bool(RE_CARD.search(text)),
        "iban": bool(RE_IBAN.search(text)),
    }


def looks_like_injection(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in INJECTION_PATTERNS)


def contains_path_traversal(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in PATH_TRAVERSAL_PATTERNS)


def links_not_allowed(text: str) -> bool:
    urls = re.findall(r"https?://([^/\s]+)", text, re.IGNORECASE)
    return any(domain.lower() not in ALLOWED_DOMAINS for domain in urls)


def scrub_user_input(user: str) -> str:
    out = re.sub(
        r"(?i)(ignore (all|previous|above) instructions|reveal .* prompt|jailbreak|act as developer)",
        "[REMOVED]",
        user
    )
    out = re.sub(r"(?i)\[system\].*?\[/system\]", "", out)
    return out.strip()


def check_guardrails(user_input: str) -> Tuple[bool, str, Dict[str, any]]:
    flags = {
        "pii": contains_pii(user_input),
        "injection": looks_like_injection(user_input),
        "path_traversal": contains_path_traversal(user_input),
        "disallowed_links": links_not_allowed(user_input),
    }

    if flags["injection"]:
        return True, "Blocked: Potential prompt injection detected", flags

    if flags["path_traversal"]:
        return True, "Blocked: Path traversal attempt detected", flags

    if flags["disallowed_links"]:
        return True, "Blocked: Disallowed external links detected", flags

    # Flag PII but don't block (just warn)
    if any(flags["pii"].values()):
        print(f" > [WARNING]: PII detected in input: {flags['pii']}")

    return False, "OK", flags