from lxml import etree
import json
import os
import gzip
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def parse_xml_and_write_jsonl(xml_file, output_jsonl_path):
    if os.path.exists(output_jsonl_path):
        return
    
    with open(output_jsonl_path, 'a', encoding='utf-8') as jsonl_file:
         # f.write("This is a new line.\n")
            
        # Efficiently stream through the XML file
        for event, elem in etree.iterparse(xml_file, tag="PubmedArticle", recover=True):
            pmid = elem.findtext(".//PMID")
            title = elem.findtext(".//ArticleTitle") or ""
            abstract_parts = elem.findall(".//AbstractText")
            abstract = " ".join(part.text for part in abstract_parts if part is not None and part.text)
            if abstract == "":
                continue

            # Create the dictionary to write as JSONL
            article = {
                "id": pmid,
                "title": title,
                "abstract": abstract
            }

            # Write the article as a JSON object on a single line
            jsonl_file.write(json.dumps(article) + "\n")

            # Clean up to free memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
                
def iter_dir(path_to_zips, output_jsonl_path):
    
    # TODO: remove corpus.jsonl if exists

    # Iterate through all zip files in the folder
    for filename in os.listdir(path_to_zips):
        if filename.endswith('.gz'):
            gz_path = os.path.join(path_to_zips, filename)

            # Open and read the gzipped XML file
            with gzip.open(gz_path, 'rb') as xml_file:
                # xml_content = xml_file.read()
                parse_xml_and_write_jsonl(xml_file, output_jsonl_path)
    

# Example usage
iter_dir("../data/baseline_raw", "../data/out/corpus.jsonl")
# parse_and_write_jsonl("../data/pubmed25n0001.xml", "../data/out/output_articles.jsonl")
