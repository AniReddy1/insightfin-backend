import os
import json
import requests
import pandas as pd
import yfinance as yf
from io import BytesIO
from zipfile import ZipFile
from groq import Groq
import google.generativeai as genai
from supabase import create_client, Client

# --- SETUP ---
NEWSDATA_API_KEY = os.environ.get("NEWSDATA_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def fetch_live_price(ticker_symbol):
    print(f"[2] Fetching live price for {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            return "₹1,024.50" # Fallback if market closed or ticker offline
        return f"₹{hist['Close'].iloc[-1]:.2f}"
    except Exception as e:
        print(f"    -> Error fetching price: {e}")
        return "₹1,024.50"

def fetch_gdelt_events():
    print("[3] Scanning GDELT Live Feed for Global Macro Events...")
    try:
        url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
        res = requests.get(url, timeout=10)
        latest_file_url = res.text.split('\n')[0].split(' ')[2]
        
        print(f"    -> Downloading latest batch: {latest_file_url.split('/')[-1]}")
        zip_res = requests.get(latest_file_url, timeout=15)
        zip_file = ZipFile(BytesIO(zip_res.content))
        csv_name = zip_file.namelist()[0]
        
        df = pd.read_csv(zip_file.open(csv_name), sep='\t', header=None, dtype=str)
        df.columns = ["GlobEventID", "Day", "MonthYear", "Year", "FractionDate", "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code", "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code", "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"]
        
        df['NumArticles'] = pd.to_numeric(df['NumArticles'], errors='coerce')
        df['GoldsteinScale'] = pd.to_numeric(df['GoldsteinScale'], errors='coerce')
        
        filtered = df[(df['ActionGeo_CountryCode'].isin(['IN', 'US'])) & (df['NumArticles'] > 10)]
        
        # Safety check if filtering returns empty dataframe
        if filtered.empty:
            filtered = df.sort_values(by='NumArticles', ascending=False)
            
        if filtered.empty:
            return {"title": "No major events in this window.", "score": 0.0, "location": "Global"}
            
        top = filtered.iloc[0]
        return {
            "title": f"Event involving {top['Actor1Name']} and {top['Actor2Name']}",
            "score": top['GoldsteinScale'],
            "location": top['ActionGeo_FullName']
        }
    except Exception as e:
        print(f"    -> Error fetching GDELT data: {e}")
        return {"title": "Global Data Unavailable", "score": 0.0, "location": "N/A"}

def fetch_company_news(company_name):
    print(f"[4] Fetching latest news for {company_name}...")
    if not NEWSDATA_API_KEY:
        return "Simulated news: Tata Motors announces new EV strategy."
    
    try:
        url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={company_name}&language=en"
        res = requests.get(url, timeout=10).json()
        articles = res.get('results', [])
        if not articles:
            return "No recent news found."
        print(f"    -> Found {len(articles)} live articles.")
        return " ".join([a.get('title', '') for a in articles[:5]])
    except Exception as e:
        print(f"    -> Error fetching news: {e}")
        return "Error retrieving news."

def get_groq_sentiment(news_text):
    print("[5] Tagging article sentiment with Groq Llama 3.1...")
    default_res = {"sentiment": "Neutral", "sentiment_score": 50, "keywords": []}
    if not GROQ_API_KEY:
        return default_res
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"Analyze this news: {news_text}. Respond ONLY with a valid JSON object containing 'sentiment' (Bullish/Bearish/Neutral), 'sentiment_score' (0-100), and 'keywords' (list of 3 strings). No markdown formatting."
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant"
        )
        res_text = chat.choices[0].message.content.strip()
        
        # Clean up any potential markdown formatting the AI might add
        if res_text.startswith("```json"):
            res_text = res_text.replace("```json", "").replace("```", "").strip()
        elif res_text.startswith("```"):
            res_text = res_text.replace("```", "").strip()
            
        return json.loads(res_text)
    except Exception as e:
        print(f"    -> Error parsing Groq: {e}")
        return default_res

def synthesize_with_gemini(macro, micro, company):
    print(f"[6] Writing final Sovereign Synthesis for {company} using Gemini Flash...")
    if not GEMINI_API_KEY:
        return "Simulated AI synthesis due to missing key."
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using gemini-1.5-flash-latest to avoid 404 versioning errors
        # model = genai.GenerativeModel('gemini-1.5-flash-latest')
        model = genai.GenerativeModel('gemini-flash-latest')
        prompt = f"Write a 3-sentence sovereign synthesis for {company}. Macro event: {macro}. Micro news: {micro}."
        res = model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        print(f"    -> Gemini Error: {e}")
        return "Analysis currently unavailable due to AI synthesis timeout."

def run_pipeline():
    print("==================================================")
    print("--- STARTING MUDRA AI PIPELINE ---")
    print("==================================================")
    
    target_company = "TATA MOTORS"
    ticker = "TATAMOTORS.NS"
    
    price = fetch_live_price(ticker)
    macro_event = fetch_gdelt_events()
    news = fetch_company_news(target_company)
    groq_tags = get_groq_sentiment(news)
    
    # Safe dictionary access (.get) prevents KeyError if the JSON was malformed
    sentiment = groq_tags.get('sentiment', 'Neutral')
    score = groq_tags.get('sentiment_score', 50)
    
    synthesis = synthesize_with_gemini(macro_event, news, target_company)
    
    print("[7] Saving to Supabase...")
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            data = {
                "ticker": ticker,
                "price": price,
                "sentiment": sentiment,
                "sentiment_score": score,
                "synthesis": synthesis
            }
            supabase.table("analyses").insert(data).execute()
            print("    -> SUCCESS: Data securely saved!")
        except Exception as e:
            print(f"    -> Supabase Error: {e}")
    else:
        print("    -> SKIPPED: Supabase keys missing.")

if __name__ == "__main__":
    run_pipeline()
