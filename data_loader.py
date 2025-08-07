import requests
from bs4 import BeautifulSoup
import json
import csv
from io import StringIO


def scrape_html(response_text: str, url: str) -> list[str]:
    """Parses HTML text and extracts clean text chunks."""
    
    soup = BeautifulSoup(response_text, 'html.parser')
    for script_or_style in soup(['script', 'style', 'nav', 'footer', 'aside']):
        script_or_style.decompose()
    
    text_chunks = [element.get_text(separator=' ', strip=True) for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span']) if element.get_text(strip=True) and len(element.get_text(strip=True)) > 25]
    print(f"Successfully parsed HTML from {url}, found {len(text_chunks)} chunks.")
    return text_chunks

def parse_json(response_text: str, url: str) -> list[str]:
    """Parses JSON text and extracts all string values."""
    try:
        data = json.loads(response_text)
        text_chunks = []
        def extract_strings(element):
            if isinstance(element, dict):
                for key, value in element.items():
                    if isinstance(value, str):
                        text_chunks.append(f"{key}: {value}")
                    else:
                        extract_strings(value)
            elif isinstance(element, list):
                for item in element:
                    extract_strings(item)
            elif isinstance(element, str) and len(element) > 10:
                text_chunks.append(element)

        extract_strings(data)
        print(f"Successfully parsed JSON from {url}, found {len(text_chunks)} chunks.")
        return text_chunks
    except json.JSONDecodeError:
        print(f"Could not parse JSON from {url}")
        return []

def parse_csv(response_text: str, url: str) -> list[str]:
    """Parses CSV text and converts each row into a text chunk."""
    text_chunks = []
    csv_file = StringIO(response_text)
    reader = csv.reader(csv_file)
    header = next(reader, None)
    
    if not header: return []
    
    for row in reader:
        if row:
            row_text = ", ".join([f"{header[i].strip()}: {row[i].strip()}" for i in range(min(len(header), len(row)))])
            text_chunks.append(row_text)
            
    print(f"Successfully parsed CSV from {url}, found {len(text_chunks)} chunks.")
    return text_chunks

def load_from_source(source: str) -> list[str]:
    """Detects the source type  and uses the appropriate parser."""
    
    if source.lower().startswith('http'):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(source, headers=headers, timeout=15)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                return parse_json(response.text, source)
            elif 'text/html' in content_type:
                return scrape_html(response.text, source)
            elif 'text/csv' in content_type:
                return parse_csv(response.text, source)
            else:
                print(f"Unknown Content-Type '{content_type}'.")
                return scrape_html(response.text, source)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {source}: {e}")
            return []

    elif source.lower().endswith('.pdf'):
        from create_embed_doc import extract_text 
        return extract_text(source)
        
    else:
        print(f"Unsupported source format: {source}")
        return []