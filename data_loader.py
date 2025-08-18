import requests
from bs4 import BeautifulSoup
import json, re
import csv
import io
import os
from io import StringIO
import mimetypes
import yfinance as yf
import google.generativeai as genai
from jira import JIRA
import numpy as np
import faiss
import pickle
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv
from create_embed_doc import extract_text, embed_chunks
import requests
# ------------------------
# DRIVE FILE HANDLING
# ------------------------

def add_drive_chunks_to_index(file_id, faiss_index_path, metadata_path, title="Google Drive Doc"):
    creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive.readonly'])
    service = build('drive', 'v3', credentials=creds)
    file_meta = service.files().get(fileId=file_id).execute()
    file_name = file_meta['name']

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    # Save to gdrive_files directory
    save_dir = os.path.join(os.getcwd(), "gdrive_files")
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, file_name)
    with open(local_path, "wb") as f:
        f.write(fh.getvalue())

    # Determine file type and parse
    ext = file_name.lower()
    if ext.endswith(".pdf"):
        new_chunks = extract_text(local_path)

    elif ext.endswith(".html") or ext.endswith(".htm"):
        html_content = fh.getvalue().decode("utf-8", errors="ignore")
        new_chunks = scrape_html(html_content, file_name)

    elif ext.endswith(".csv"):
        csv_content = fh.getvalue().decode("utf-8", errors="ignore")
        new_chunks = parse_csv(csv_content, file_name)

    elif ext.endswith(".json"):
        json_content = fh.getvalue().decode("utf-8", errors="ignore")
        new_chunks = parse_json(json_content, file_name)

    else:
        print(f"Unsupported file type: {file_name}")
        return

    if not new_chunks:
        print(f"No chunks extracted from {file_name}")
        return

    # Embed and add to FAISS
    new_embeddings = embed_chunks(new_chunks, title=title)
    index = faiss.read_index(faiss_index_path)
    with open(metadata_path, 'rb') as f:
        all_chunks = pickle.load(f)

    index.add(np.array(new_embeddings).astype('float32'))
    all_chunks.extend(new_chunks)

    faiss.write_index(index, faiss_index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_chunks, f)

    print(f"Added {len(new_chunks)} chunks from {file_name} to FAISS index.")
    
    
# ------------------------
# YAHOO FINANCE LIVE DATA
# ------------------------ 

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json"
}


def fetch_yahoo_screener(scr_id, count=5):
    """Helper to fetch Yahoo screener data safely."""
    url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count={count}&scrIds={scr_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        quotes = data["finance"]["result"][0]["quotes"]
        return [
            f"{q['symbol']} ({q.get('shortName','')}): {round(q['regularMarketChangePercent'],2)}%"
            for q in quotes
        ]
    except Exception as e:
        return [f"⚠️ Could not fetch {scr_id}: {e}"]

def get_top_gainers_losers_global(count=5):
    gainers = fetch_yahoo_screener("day_gainers", count)
    losers = fetch_yahoo_screener("day_losers", count)
    return gainers, losers

def get_top_gainers_losers_india(count=5):
    gainers = fetch_yahoo_screener("in_top_gainers", count)
    losers = fetch_yahoo_screener("in_top_losers", count)
    return gainers, losers

    
    
def interpret_yahoo_query_with_llm(user_question):
    
    system_prompt = """
    You are a financial data extraction assistant.
    From the user's question, extract:
    - stock ticker symbol (e.g., TSLA for Tesla)
    - period (e.g., 1y, 6mo, 3mo, 5y, max)
    - interval (e.g., 1d, 1wk, 1mo)

    Rules:
    - Map company names to their Yahoo tickers (Tesla→TSLA, Apple→AAPL, Microsoft→MSFT, Google→GOOG, Amazon→AMZN, Meta→META).
    - If unclear, default to 1y period and 1d interval.
    - Respond ONLY with valid JSON and nothing else.
    Example: {"symbol": "TSLA", "period": "1y", "interval": "1d"}
    """

    print(f"DEBUG: User question: {user_question}")  # DEBUG
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(f"{system_prompt}\nUser question: {user_question}")

    # Clean and extract JSON
    raw_text = response.text.strip()
    print(f"DEBUG: Raw LLM response: '{raw_text}'")  # DEBUG 
    
    raw_text = re.sub(r"```(json)?", "", raw_text)
    raw_text = raw_text.strip("` \n")
    print(f"DEBUG: Cleaned text: '{raw_text}'")  # DEBUG

    try:
        params = json.loads(raw_text)
        print(f"DEBUG: Successfully parsed params: {params}")  # DEBUG
        return params
    except Exception as e:
        print(f"DEBUG: JSON parsing failed: {e}")  # DEBUG
        print(f"DEBUG: Raw output was: '{raw_text}'")  # DEBUG

        # Fallback
        
        print("DEBUG: Using fallback detection")  # DEBUG
        if "tesla" in user_question.lower():
            print("DEBUG: Tesla detected in fallback")  # DEBUG
            return {"symbol": "TSLA", "period": "6mo", "interval": "1d"}
        elif "apple" in user_question.lower():
            return {"symbol": "AAPL", "period": "6mo", "interval": "1d"}
        else:
            print("DEBUG: No company detected in fallback")  # DEBUG
            return None
        
def fetch_yahoo_data(symbol, period, interval):
    """
    Fetch historical market data from Yahoo Finance.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"Yahoo fetch error: {e}")
        return None

def yahoo_llm_answer(user_question, generate_final_answer_fn):
    q_lower = user_question.lower()

    if "india" in q_lower or "nse" in q_lower or "bse" in q_lower:
        gainers, losers = get_top_gainers_losers_india()
        context = (
            "📈 Top Gainers Today (India):\n" + "\n".join(gainers) +
            "\n\n📉 Top Losers Today (India):\n" + "\n".join(losers)
        )
        return generate_final_answer_fn(user_question, [context])

    if "gainer" in q_lower or "loser" in q_lower:
        gainers, losers = get_top_gainers_losers_global()
        context = (
            "📈 Top Gainers Today (Global):\n" + "\n".join(gainers) +
            "\n\n📉 Top Losers Today (Global):\n" + "\n".join(losers)
        )
        return generate_final_answer_fn(user_question, [context])

    # fallback: ticker query
    params = interpret_yahoo_query_with_llm(user_question)
    if not params or "symbol" not in params or not params["symbol"]:
        return "I couldn't figure out which stock you meant. Please specify a ticker or company name."

    hist = fetch_yahoo_data(params["symbol"], params["period"], params["interval"])
    if hist is None:
        return f"No data found for {params['symbol']}."

    context = f"Stock data for {params['symbol']} ({params['period']}, {params['interval']}):\n"
    context += hist.tail(100).to_string()
    return generate_final_answer_fn(user_question, [context])




# ------------------------
# JIRA LIVE DATA
# ------------------------
load_dotenv()

def get_jira_client():
    jira_url = "https://thesunrise.atlassian.net/"
    api_token = os.getenv("JIRA_API_TOKEN")
    email = os.getenv("EMAIL")
    if not api_token or not email:
        raise ValueError("JIRA_API_TOKEN and EMAIL must be set in .env file")
    return JIRA(
        server=jira_url,
        basic_auth=(email, api_token)
    )

def interpret_jira_query_with_llm(user_question):
    """
    Ask the LLM to decide what Jira data to fetch.
    """
    system_prompt = """
    You are a Jira data extraction assistant.
    From the user's question, decide what Jira entity to fetch.

    Supported types: issues_by_project, issue_status, sprint_progress, assignee_tasks.

    Notes for interpretation:
    - The Jira project "Momentum" always uses key MTM (not MOM).
    - Status synonyms:
      "To Do" → ["To Do","Open","Backlog"]
      "Ongoing" → ["Ongoing","In Progress"]
      "Done" → ["Done","Closed","Resolved"]
    - Always include "status" if the user specifies one.
    - Always include "assignee" if the user specifies one.
    - Respond ONLY in valid JSON.

    Example:
    {"type": "issues_by_project", "project_key": "MTM", "status": "To Do"}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(f"{system_prompt}\nUser question: {user_question}")

    raw_text = response.text if hasattr(response, 'text') else str(response)
    print(f"DEBUG: Raw Jira LLM response: '{raw_text}'")

    cleaned_text = raw_text.strip()
    if cleaned_text.startswith('```'):
        lines = cleaned_text.split('\n')
        cleaned_text = '\n'.join(lines[1:-1])
    cleaned_text = cleaned_text.strip('`').strip()
    print(f"DEBUG: Cleaned Jira text: '{cleaned_text}'")

    try:
        params = json.loads(cleaned_text)
        print(f"DEBUG: Parsed Jira params: {params}")

        # --- Project key normalization ---
        PROJECT_KEY_MAP = {
            "momentum": "MTM",
            "proj momentum": "MTM",
            "mom": "MTM"
        }
        user_key = params.get("project_key", "").lower()
        if user_key in PROJECT_KEY_MAP:
            params["project_key"] = PROJECT_KEY_MAP[user_key]

        return params
    except Exception as e:
        print(f"Error parsing Jira LLM output: {e}")
        print(f"DEBUG: Attempted to parse: '{cleaned_text}'")
        return None


def fetch_jira_data(params):
    jira = get_jira_client()

    # --- Status normalization ---
    STATUS_MAP = {
        "to do": ["To Do", "Open", "Backlog"],
        "ongoing": ["Ongoing", "In Progress"],
        "done": ["Done", "Closed", "Resolved"]
    }

    if params["type"] == "issues_by_project":
        jql = f'project={params["project_key"]}'

        # --- Status filter ---
        if "status" in params:
            user_status = params["status"].lower()
            if user_status in STATUS_MAP:
                status_list = ",".join([f'"{s}"' for s in STATUS_MAP[user_status]])
                jql += f" AND status in ({status_list})"
            else:
                jql += f' AND status="{params["status"]}"'

        # --- Assignee filter ---
        if "assignee" in params and params["assignee"]:
            # If it's explicitly None or empty → skip filtering
            if params["assignee"].lower() not in ["none", "null", ""]:
                jql += f' AND assignee="{params["assignee"]}"'
            else:
                print("DEBUG: Assignee was null → skipping filter but will display names.")

        print(f"DEBUG: Running JQL -> {jql}")
        issues = jira.search_issues(jql, maxResults=20)

        if not issues:
            return f"No issues found in project {params['project_key']} with filters."

        result_lines = []
        for i in issues:
            assignee = i.fields.assignee.displayName if i.fields.assignee else "Unassigned"
            reporter = i.fields.reporter.displayName if i.fields.reporter else "Unknown"
            created = i.fields.created.split("T")[0] if hasattr(i.fields, "created") else "N/A"
            result_lines.append(
                f"{i.key} - {i.fields.summary} - {i.fields.status.name} "
                f"- Assignee: {assignee} - Reporter: {reporter} - Created: {created}"
            )
        return "\n".join(result_lines)


    elif params["type"] == "issue_status":
        issue = jira.issue(params["issue_key"])
        assignee = issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned"
        return f"Issue {issue.key}: {issue.fields.summary}, Status: {issue.fields.status.name}, Assignee: {assignee}"

    elif params["type"] == "sprint_progress":
        return "Sprint progress retrieval not fully implemented yet."

    elif params["type"] == "assignee_tasks":
        jql = f'assignee="{params["assignee"]}" AND status!="Done"'
        issues = jira.search_issues(jql, maxResults=20)
        if not issues:
            return f"No tasks found for assignee {params['assignee']}."
        return "\n".join([
            f"{i.key} - {i.fields.summary} - {i.fields.status.name}"
            for i in issues
        ])

    return "Unsupported Jira query."



def jira_llm_answer(user_question, generate_final_answer_fn):
    params = interpret_jira_query_with_llm(user_question)
    if not params:
        return "Could not interpret Jira request."

    jira_data = fetch_jira_data(params)
    return generate_final_answer_fn(user_question, [jira_data])

# ------------------------
# TEXT PARSING FUNCTIONS
# ------------------------

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