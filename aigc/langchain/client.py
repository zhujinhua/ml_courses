import requests
import json

response = requests.post(
    "http://localhost:8001/intention_rec/invoke",
    json={"input": {"role": "意图识别专家", "content": "荨麻疹常见症状？"}}
)
print(json.loads(response.json()['output']['content']))

