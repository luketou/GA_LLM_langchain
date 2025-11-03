import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(
    # This is the default and can be omitted
    api_key="csk-r45fve4re56cxcmmvj8j4f5c5hrvt893deyk9p9pwre36t38"
)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "今天天氣如何"
        }
    ],
    model="llama-4-scout-17b-16e-instruct",
    stream=True,
    max_completion_tokens=2048,
    temperature=0.2,
    top_p=1
)

for chunk in stream:
  print(chunk.choices[0].delta.content or "", end="")