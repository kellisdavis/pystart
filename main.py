import os
import sys

import groq
from groq import Groq


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Missing environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def main() -> None:
    require_env("GROQ_API_KEY")

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = Groq()

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    print('Groq CLI Chat — type "quit" to exit.\n')

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_text:
            continue

        if user_text.lower() == "quit":
            print("Goodbye.")
            break

        messages.append({"role": "user", "content": user_text})

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
        except groq.AuthenticationError:
            print("Auth failed. Check GROQ_API_KEY.")
            sys.exit(1)
        except groq.APIError as e:
            print(f"Groq API error: {e}")
            messages.pop()
            continue

        answer = completion.choices[0].message.content or ""
        print(f"\nAssistant: {answer}\n")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
