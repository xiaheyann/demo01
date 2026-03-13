from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:58000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="model-bin/xin7697/Qwen3-14B-W8A8-Ascend",
    messages=[
        {"role": "user", "content": "你好，简单介绍一下你自己。"}
    ],
    temperature=0.0,
    max_tokens=64,
)

print(resp.choices[0].message.content)