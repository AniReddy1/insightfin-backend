import os
import sys
import time
import json
import argparse
import requests
import pandas as pd
import yfinance as yf
from io import BytesIO
from zipfile import ZipFile
from groq import Groq
import google.generativeai as genai
from supabase import create_client, Client

# --- ENVIRONMENT KEYS ---
NEWSDATA_API_KEY = os.environ.get("NEWSDATA_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def resolve_ticker_smart(query):
    """
    Dynamically resolves any user input (symbol or company name).
    Priority:
      1. Indian Equity (NSE: .NS, BSE: .BO)
      2. US Equity (NASDAQ, NYSE, etc.)
      3. Discard / Ignore if invalid
    """
    query = query.strip()
    print(f"    -> Resolving identity for '{query}'...")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=6&newsCount=0"
    
    try:
        res = requests.get(search_url, headers=headers, timeout=8).json()
        quotes = res.get('quotes', [])
        
        if not quotes:
            print(f"    [-] No market listings found for '{query}'.")
            return None

        # Filter only EQUITY quotes
        equities = [q for q in quotes if q.get('quoteType') == 'EQUITY']
        if not equities:
            equities = quotes

        selected_quote = None

        # Priority 1: Check Indian Markets (NSE / BSE)
        for q in equities:
            symbol = q.get('symbol', '')
            exchange = q.get('exchange', '').upper()
            if symbol.endswith('.NS') or symbol.endswith('.BO') or exchange in ['NSI', 'BSE', 'NSE']:
                selected_quote = q
                break

        # Priority 2: Check US / Global Markets
        if not selected_quote:
            for q in equities:
                exchange = q.get('exchange', '').upper()
                if exchange in ['NMS', 'NYQ', 'NGM', 'NASDAQ', 'NYSE', 'PCX']:
                    selected_quote = q
                    break

        # Fallback to the top search result if neither explicit rule matched
        if not selected_quote and equities:
            selected_quote = equities[0]

        if not selected_quote:
            return None

        resolved_symbol = selected_quote.get('symbol')
        company_name = selected_quote.get('shortname') or selected_quote.get('longname') or query
        exchange_name = selected_quote.get('exchange', 'GLOBAL')
# Fetch live market data natively using requests to bypass GitHub Actions IP block
        print(f"    -> Fetching live market data natively for {resolved_symbol}...")
        chart_url = f"https://query2.finance.yahoo.com/v8/finance/chart/{resolved_symbol}?interval=1d&range=5d"
        
        chart_res = requests.get(chart_url, headers=headers, timeout=10)
        
        try:
            chart_data = chart_res.json()
            result = chart_data.get('chart', {}).get('result')
            
            if not result:
                print(f"    [-] No chart result returned for '{resolved_symbol}'.")
                return None
                
            # Extract closing prices and filter out None values (which happen on market holidays/halts)
            closes = result[0].get('indicators', {}).get('quote', [{}])[0].get('close', [])
            valid_closes = [c for c in closes if c is not None]
            
            if len(valid_closes) < 1:
                print(f"    [-] No valid closing prices found for '{resolved_symbol}'.")
                return None
                
            close_price = valid_closes[-1]
            prev_close = valid_closes[-2] if len(valid_closes) > 1 else close_price
            change_pct = ((close_price - prev_close) / prev_close) * 100
            
        except Exception as e:
            print(f"    [-] Native price fetch error for '{resolved_symbol}': {e}")
            return None
        
        currency = "INR" if (".NS" in resolved_symbol or ".BO" in resolved_symbol) else "USD"
        curr_sign = "₹" if currency == "INR" else "$"
        
        return {
            "symbol": resolved_symbol,
            "company_name": company_name,
            "exchange": exchange_name,
            "price": f"{curr_sign}{close_price:,.2f}",
            "change": f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%"
        }

    except Exception as e:
        print(f"    [-] Ticker resolution error for '{query}': {e}")
        return None

def fetch_gdelt_macro():
    """Fetches global macro pulse once per run."""
    print("[1] Scanning GDELT Global Macro Stream...")
    try:
        url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
        res = requests.get(url, timeout=10)
        latest_file_url = res.text.split('\n')[0].split(' ')[2]
        
        zip_res = requests.get(latest_file_url, timeout=15)
        zip_file = ZipFile(BytesIO(zip_res.content))
        csv_name = zip_file.namelist()[0]
        
        df = pd.read_csv(zip_file.open(csv_name), sep='\t', header=None, dtype=str)
        df.columns = ["GlobEventID", "Day", "MonthYear", "Year", "FractionDate", "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code", "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code", "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"]
        
        df['NumArticles'] = pd.to_numeric(df['NumArticles'], errors='coerce')
        filtered = df[df['NumArticles'] > 10]
        if not filtered.empty:
            top = filtered.iloc[0]
            return f"Global event between {top.get('Actor1Name', 'Market')} and {top.get('Actor2Name', 'Trade Partners')}."
    except Exception as e:
        print(f"    -> GDELT Notice: {e}")
    return "Stable macro trading conditions."

def fetch_company_news(company_name):
    """Fetches real-time news headlines."""
    print(f"    -> Fetching live headlines for {company_name}...")
    if not NEWSDATA_API_KEY:
        return "Market volume indicates standard institutional trading."
    try:
        url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={company_name}&language=en"
        res = requests.get(url, timeout=10).json()
        articles = res.get('results', [])
        if articles:
            return " ".join([a.get('title', '') for a in articles[:4]])
    except Exception as e:
        print(f"    -> News retrieval notice: {e}")
    return f"Active trading observed for {company_name}."

def get_groq_sentiment(news_text):
    """Scores news sentiment using Groq Llama 3.1."""
    print("    -> Scoring market sentiment with Groq Llama 3.1...")
    default_res = {"sentiment": "Neutral", "sentiment_score": 50}
    if not GROQ_API_KEY:
        return default_res
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"Analyze this market news: {news_text}. Return ONLY a JSON object with 'sentiment' (Bullish/Bearish/Neutral) and 'sentiment_score' (0-100 integer). No markdown formatting."
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        res_text = chat.choices[0].message.content.strip()
        if res_text.startswith("```json"):
            res_text = res_text.replace("```json", "").replace("```", "").strip()
        elif res_text.startswith("```"):
            res_text = res_text.replace("```", "").strip()
        return json.loads(res_text)
    except Exception as e:
        print(f"    -> Sentiment notice: {e}")
        return default_res

def synthesize_with_gemini(macro, news, name, price, exchange):
    """Produces a sovereign financial synthesis using Gemini Flash."""
    print(f"    -> Writing sovereign synthesis for {name} with Gemini Flash...")
    if not GEMINI_API_KEY:
        return f"{name} demonstrates steady fundamentals on {exchange}."
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-flash-latest')
        prompt = (
            f"You are a sovereign financial intelligence analyst. Provide a concise 3-sentence synthesis for {name} "
            f"(Listed on: {exchange}, Price: {price}). Macro background: {macro}. News context: {news}. "
            f"Focus on catalysts, risk factors, and momentum."
        )
        res = model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        print(f"    -> Gemini notice: {e}")
        return f"{name} shows consistent volume and institutional interest."

def main():
    parser = argparse.ArgumentParser(description="InsightFin AI Pipeline")
    parser.add_argument("--ticker", type=str, help="Specific ticker or company name to analyze")
    args = parser.parse_args()

    print("==================================================")
    print("--- STARTING DYNAMIC SOVEREIGN AI ENGINE ---")
    print("==================================================")

    # 1. Initialize Supabase client
    supabase: Client = None
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"[-] Supabase init error: {e}")

# 2. Collect targets strictly from dynamic sources
    tasks = [] # Will hold dicts: {"id": queue_id, "query": search_term}
    
    if args.ticker:
        tasks.append({"id": None, "query": args.ticker})
        print(f"[+] Direct CLI Input received: {args.ticker}")
    elif supabase:
        try:
            reqs = supabase.table("search_requests").select("*").eq("status", "pending").limit(10).execute()
            for row in (reqs.data or []):
                tasks.append({"id": row['id'], "query": row['ticker']})
            
            if tasks:
                print(f"[+] Loaded {len(tasks)} user requests from Supabase search queue.")
        except Exception as e:
            print(f"[-] Queue read notice: {e}")

    if not tasks:
        print("[!] No pending requests or CLI tickers found. Pipeline shutting down cleanly.")
        return

    # 3. Fetch global macro pulse once
    macro_event = fetch_gdelt_macro()

    # 4 & 5 & 6. Process, Save, and Update Queue per item
    for task in tasks:
        raw_query = task["query"]
        q_id = task["id"]
        
        print(f"\n>>> Processing Query: '{raw_query}'")
        
        asset = resolve_ticker_smart(raw_query)
        if not asset:
            print(f"[!] '{raw_query}' could not be resolved. Skipping.")
            # Mark as failed so the frontend doesn't wait forever
            if supabase and q_id:
                supabase.table("search_requests").update({"status": "failed"}).eq("id", q_id).execute()
            continue

        print(f"    [✓] Matched: {asset['company_name']} ({asset['symbol']}) on {asset['exchange']} at {asset['price']}")
        
        # Pipeline execution
        news = fetch_company_news(asset['company_name'])
        sentiment_data = get_groq_sentiment(news)
        synthesis = synthesize_with_gemini(
            macro_event, news, asset['company_name'], asset['price'], asset['exchange']
        )

        if supabase:
            # Upsert into analyses (prevents duplication errors)
            try:
                record = {
                    "ticker": asset['symbol'],
                    "price": asset['price'],
                    "sentiment": sentiment_data.get('sentiment', 'Neutral'),
                    "sentiment_score": sentiment_data.get('sentiment_score', 50),
                    "synthesis": synthesis
                }
                # Using upsert instead of insert handles subsequent requests for the same stock
                supabase.table("analyses").upsert(record, on_conflict="ticker").execute()
                print(f"    [✓] Saved analysis for {asset['symbol']} to Supabase.")
            except Exception as e:
                print(f"    [-] Database write error: {e}")
        # ---> ADD THE PAUSE HERE <---
        # This pauses the loop for 5 seconds before moving to the next stock 
        # to prevent Gemini from throwing a 429 Rate Limit error.
        print("    -> Pausing for 5 seconds to respect API limits...")
        time.sleep(5)    
            # Update the specific queue row with the resolved_ticker
            if q_id:
                try:
                    supabase.table("search_requests").update({
                        "status": "completed",
                        "resolved_ticker": asset['symbol']
                    }).eq("id", q_id).execute()
                    print(f"    [+] Updated queue ID {q_id} to completed with ticker {asset['symbol']}.")
                except Exception as e:
                    print(f"    [-] Queue update notice: {e}")
if __name__ == "__main__":
    main()
