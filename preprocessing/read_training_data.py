import json
import requests

API_URL = "http://bioasq.org:8000/"

# Retrieve the session key from the API
def get_session_key() -> str:
    response = requests.get(f"{API_URL}pubmed/")
    session_key = response.text.split("/")[-1]
    return session_key

# Retrieve documents from API of a given query
def retrieve_documents_from_api(query: str, session_key: str, page = 0, articles_per_page = 10) -> list:
    payload = {
        "findPubMedCitations": [query, page, articles_per_page]
    }
    json_string = json.dumps(payload)

    response = requests.post(
        url=f"{API_URL}/{session_key}", 
        data = {"json": json_string}
    )
    documents = []
    for document in response.json()['result']['documents']:
        if not all(k in document for k in ("pmid", "title", "documentAbstract")):
            continue
        
        documents.append({
            "id": document["pmid"],
            "title": document["title"],
            "abstract": document["documentAbstract"],
        })   
    return documents


# Setup training data questions
with open("data/training13b.json", "r") as f:
    data = json.load(f)
questions = {q["id"]: {"query": q["body"], "relevant_documents": q["documents"]} for q in data["questions"]}
