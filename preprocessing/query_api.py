import requests
from xml.etree import ElementTree as ET

def query_pubmed(term, max_results=10):
    # Base URLs
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    
    # Step 1: Use ESearch to find PubMed IDs (PMIDs)
    params_esearch = {
        "db": "pubmed",
        "term": term,
        "retmax": max_results,
        "retmode": "xml"
    }
    print(f"Searching PubMed for: {term}")
    response = requests.get(esearch_url, params=params_esearch)
    response.raise_for_status()
    
    root = ET.fromstring(response.text)
    pmids = [id_elem.text for id_elem in root.findall(".//Id")]
    
    if not pmids:
        print("No results found.")
        return

    # Step 2: Use ESummary to get article details
    params_esummary = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }
    response = requests.get(esummary_url, params=params_esummary)
    response.raise_for_status()
    
    root = ET.fromstring(response.text)

    # Step 3: Print the results
    for docsum in root.findall(".//DocSum"):
        title = ""
        authors = []
        pubdate = ""
        
        for item in docsum.findall("Item"):
            if item.attrib["Name"] == "Title":
                title = item.text
            elif item.attrib["Name"] == "AuthorList":
                authors = [child.text for child in item.findall("Item")]
            elif item.attrib["Name"] == "PubDate":
                pubdate = item.text

        print(f"\nTitle: {title}")
        print(f"Authors: {', '.join(authors)}")
        print(f"Publication Date: {pubdate}")

# Example usage
if __name__ == "__main__":
    query_pubmed("List signaling molecules (ligands) that interact with the receptor EGFR?", max_results=10)
