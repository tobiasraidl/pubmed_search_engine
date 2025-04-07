import sys
import io
from lxml import etree
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def parse_and_write_jsonl(xml_path, output_jsonl_path):
    with open(output_jsonl_path, "w") as jsonl_file:
        # Efficiently stream through the XML file
        for event, elem in etree.iterparse(xml_path, tag="PubmedArticle", recover=True):
            pmid = elem.findtext(".//PMID")
            title = elem.findtext(".//ArticleTitle") or ""
            abstract_parts = elem.findall(".//AbstractText")
            abstract = " ".join(part.text for part in abstract_parts if part is not None and part.text)

            text = f"{title.strip()} {abstract.strip()}".strip()

            # Create the dictionary to write as JSONL
            article = {
                "id": pmid,
                "contents": text
            }

            # Write the article as a JSON object on a single line
            jsonl_file.write(json.dumps(article) + "\n")

            # Clean up to free memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

# Example usage
parse_and_write_jsonl("../data/pubmed25n0001.xml", "../data/out/output_articles.jsonl")
