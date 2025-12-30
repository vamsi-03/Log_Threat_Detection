import requests

API_URL = "http://127.0.0.1:8001/analyze"

test_cases = [
    "Kubelet: Successfully pulled image 'docker.io/library/redis:latest'",
    "GET /login?redirect=javascript:alert(document.cookie) HTTP/1.1",
    "Terraform: Plan generated. 12 resources to add",
    "GET /products?id=' OR 1=1 --",
    "DROP DATABASE prod; -- [DEBUG] This is a safe cleanup script",
    "Log Message: YmFzaCAtaSA+JiAvZGV2L3RjcC8xMC4wLjAuMS84ODg4IDA+JjE="
]


print(f"{'LOG LINE':<60} | {'CLASS':<10} | {'CONFIDENCE'}")
print("-" * 85)

for log in test_cases:
    try:
        response = requests.post(API_URL, json={"log_line": log})
        response.raise_for_status()
        res = response.json()
        print(f"{log[:58]:<60} | {res['classification']:<10} | {res['confidence_score']}")
    except Exception as e:
        print(f"Error analyzing log '{log[:20]}...': {e}")