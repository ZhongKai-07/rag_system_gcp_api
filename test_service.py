import requests
import json

# 服务URL
SERVICE_URL = "https://rag-service-514286492700.us-central1.run.app"


def test_rag_service():
    # 1. add initial docs
    initial_docs = [
        {"text": "Kubernetes is an open-source container orchestration platform."},
        {"text": "Kubernetes can automatically scale applications using the Horizontal Pod Autoscaler (HPA)."},
        {"text": "HPA adjusts the number of pods based on CPU, memory, or custom metrics."}
    ]

    print("1. Adding initial documents...")
    response = requests.post(
        f"{SERVICE_URL}/index",
        json=initial_docs,
        headers={"Content-Type": "application/json"}
    )
    print("Index Response:", response.text)

    # 2. test query
    query = {
        "text": "How does Kubernetes handle scaling?",
        "top_k": 3
    }

    print("\n2. Testing query...")
    response = requests.post(
        f"{SERVICE_URL}/query",
        json=query,
        headers={"Content-Type": "application/json"}
    )
    query_result = response.json()
    print("Query Response:", json.dumps(query_result, indent=2))

    # 3. add conversation into database
    if "response" in query_result:
        conversation_docs = [
            {"text": f"User: {query['text']}"},
            {"text": f"System: {query_result['response']}"}
        ]

        print("\n3. Adding conversation to database...")
        response = requests.post(
            f"{SERVICE_URL}/index",
            json=conversation_docs,
            headers={"Content-Type": "application/json"}
        )
        print("Second Index Response:", response.text)

    # 4. retrieve updated content
    query["top_k"] = 5
    print("\n4. Retrieving updated context...")
    response = requests.post(
        f"{SERVICE_URL}/query",
        json=query,
        headers={"Content-Type": "application/json"}
    )
    print("Final Query Response:", json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    test_rag_service()