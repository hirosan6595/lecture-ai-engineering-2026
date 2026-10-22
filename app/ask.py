"""Gemini / NVIDIA / Groqに、会話履歴なしで1回だけ質問する。

例（Codespaces / Bash）:
    AIE_MODEL=nvidia:meta/llama-3.3-70b-instruct uv run python -m app.ask
    AIE_MODEL=groq:openai/gpt-oss-20b uv run python -m app.ask "LoRAとは何ですか？"

または AIE_PROVIDER=groq AIE_MODEL=openai/gpt-oss-20b のように指定する。
接頭辞もAIE_PROVIDERも省略した場合、接続先はGemini。
"""

import argparse
import math
import os
import sys


DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
PROVIDERS = {
    "gemini": (None, "GEMINI_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}


def settings() -> tuple[str, str, float]:
    model = os.getenv("AIE_MODEL", "").strip()
    if not model:
        raise ValueError("AIE_MODELを指定してください。例: groq:openai/gpt-oss-20b")

    provider = os.getenv("AIE_PROVIDER", "").strip().lower()
    if ":" in model:
        prefix, model = model.split(":", 1)
        prefix = prefix.lower()
        if provider and provider != prefix:
            raise ValueError("AIE_PROVIDERとAIE_MODELの接頭辞が一致しません。")
        provider = prefix
    provider = provider or "gemini"
    model = model.strip()
    if provider not in PROVIDERS:
        raise ValueError("接続先にはgemini / nvidia / groqを指定してください。")
    if not model:
        raise ValueError("モデル名が空です。")

    temperature = float(os.getenv("AIE_TEMPERATURE", "1.0"))
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("AIE_TEMPERATUREは0〜2の数値にしてください。")
    return provider, model, temperature


def ask(question: str) -> tuple[str | None, int | None, int | None]:
    provider, model, temperature = settings()
    base_url, key_name = PROVIDERS[provider]
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"CodespacesのSecretsに{key_name}を登録してください。")

    if provider == "gemini":
        from google import genai
        from google.genai import types

        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=model,
                contents=question,
                config=types.GenerateContentConfig(temperature=temperature),
            )
        usage = response.usage_metadata
        return (
            response.text,
            usage.prompt_token_count if usage else None,
            usage.candidates_token_count if usage else None,
        )

    from openai import OpenAI

    # SDKの自動再試行を止め、1実行につき1リクエストにする。
    with OpenAI(
        base_url=base_url, api_key=api_key, max_retries=0, timeout=180.0
    ) as client:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=8192,
        )
    usage = response.usage
    answer = response.choices[0].message.content if response.choices else None
    return (
        answer,
        usage.prompt_tokens if usage else None,
        usage.completion_tokens if usage else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini / NVIDIA / Groqに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    try:
        provider, model, temperature = settings()
        print(f"[provider] {provider}", flush=True)
        print(f"[model] {model}", flush=True)
        print(f"[temperature] {temperature}", flush=True)
        answer, input_tokens, output_tokens = ask(args.question)
    except (ValueError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        print(
            f"[API ERROR] {type(exc).__name__} status={status}", file=sys.stderr
        )
        print("接続先・モデル名・キーの権限・利用上限を確認してください。", file=sys.stderr)
        raise SystemExit(1) from None

    print(f"[tokens] input={input_tokens} output={output_tokens}")
    print(f"[質問] {args.question}\n")
    if not answer:
        print("[ERROR] 本文が空です。出力上限やモデルの設定を確認してください。", file=sys.stderr)
        raise SystemExit(1)
    print(answer)


if __name__ == "__main__":
    main()
