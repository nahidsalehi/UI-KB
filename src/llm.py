"""
Ollama LLM client.
Sends a (question, context, domain) triplet and returns the generated answer.
"""

import re
import requests
import config


class OllamaClient:
    def __init__(
        self,
        url:   str = config.LLM_URL,
        model: str = config.LLM_MODEL,
        options: dict | None = None,
    ):
        self.url     = url
        self.model   = model
        self.options = options or config.LLM_OPTIONS

    def ask(self, question: str, context: str, domain: str) -> str:
        domain_name = (
            "شیوه نامه انضباطی" if domain == "discipline" else "آیین نامه آموزشی"
        )

        system_prompt = f"""شما یک دستیار تخصصی دانشگاه هستید.

فقط بر اساس منابع داده شده پاسخ بده.

قوانین:
- فقط فارسی پاسخ بده
- پاسخ کوتاه و مستقیم باشد
- توضیح اضافی نده
- متن منابع را تکرار نکن
- اگر پاسخ در منابع نبود فقط بنویس: اطلاعات کافی در منابع موجود نیست.

نوع سند: {domain_name}"""

        user_prompt = f"منابع:\n\n{context}\n\nسوال:\n{question}"

        payload = {
            "model":   self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream":  False,
            "options": self.options,
        }

        resp = requests.post(self.url, json=payload, timeout=240)
        resp.raise_for_status()

        answer = resp.json().get("message", {}).get("content", "").strip()
        answer = re.sub(r"<.*?>", "", answer)
        answer = answer.replace("[/منبع]", "").strip()
        return answer

    def __repr__(self):
        return f"OllamaClient(model={self.model})"
