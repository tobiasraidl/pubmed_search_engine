import json
import requests


with open("../data/training13b.json", "r") as f:
    data = json.load(f)

quesitons = {q["id"]: {"query": q["body"], "relevant_documents": q["documents"]} for q in data["questions"]}



API_URL = "http://bioasq.org:8000/pubmed"

def get_session_key() -> str:
    print(requests.get(API_URL))
    
session_key = get_session_key()

api_responses = {}

for id, value in quesitons.items():
    query = value["query"]
    relevant_documents = value["relevant_documents"]
    
    
    request = {
        "keywords": value,
        "page": "0",
        "articlesPerPage": "50"
    }

    response = requests.post(f"{API_URL}?{session_key}", request)
    api_responses[id] = response.text