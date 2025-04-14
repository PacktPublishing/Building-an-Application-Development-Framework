#!/usr/bin/env python3
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def main():
    user_input = "Create a haiku"
    system_prompt = "You are a poet"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
    )
    print(f"User input: {user_input}")
    print("\nAgent answer:\n")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
