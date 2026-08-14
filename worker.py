import json
import requests
import yfinance as yf
from datetime import datetime
import zipfile
import io
import pandas as pd
import os
from supabase import create_client, Client

# Optional: To use the real AI once you have keys, you will need to pip install these:
# pip install groq google-generativeai supabase
try:
    from groq import Groq
    import google.generativeai as genai
    AI_LIBRARIES_INSTALLED = True
except ImportError:
    AI_LIBRARIES_INSTALLED = False

# GDELT uses numerical "CAMEO" codes to describe events. 
CAMEO_ROOT_MAP = {
    "01": "Public Statement", "02": "Appeal", "03": "Express Intent to Cooperate",
    "04": "Consult/Meet", "05": "Engage in Diplomatic Coop", "06": "Material Cooperation",
    "07": "Provide Aid", "08": "Yield/Concede", "09": "Investigate",
    "10": "Demand", "11": "Disapprove/Object", "12": "Reject",
    "13": "Threaten", "14": "Protest/Riot", "15": "Exhibit Military Posture",
    "16": "Reduce Relations/Sanctions", "17": "Coerce", "18": "Assault",
    "19": "Fight/Military Conflict", "20": "Mass Violence"
}

# ==========================================
# CONFIGURATION & API KEYS (SECURE CLOUD MODE)
# ==========================================
# We now use os.environ.get() so keys are pulled from Render's secure vault, 
# not hardcoded in the script. This keeps your accounts safe from hackers!
NEWSDATA_API_KEY = os.environ.get("NEWSDATA_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# ==========================================
# STEP 1: LOAD THE RULEBOOK (SECTOR MAP)
# ==========================================
def load_sector_map(filepath="sector_exposure_map.json"):
    """Reads our JSON file so the AI knows how to link events to stocks."""
    print("[1] Loading Sector Exposure Map...")
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("    -> Map not found, loading fallback rules...")
        return {"nse_sector_mapping": {}}

# ==========================================
# STEP 2: FETCH LIVE MARKET DATA (FREE)
# ==========================================
def get_live_price(ticker_symbol):
    """Uses Yahoo Finance to get live NSE prices for free."""
    print(f"[2] Fetching live price for {ticker_symbol}...")
    try:
        stock = yf.Ticker(ticker_symbol)
        # Grab the last closing price
        price = stock.history(period="1d")['Close'].iloc[-1]
        return round(price, 2)
    except Exception as e:
        print(f"    -> Error fetching price: {e}")
        return None

# ==========================================
# STEP 3: THE MACRO ENGINE (GDELT)
# ==========================================
def fetch_gdelt_events(target_country="IN"):
    """Fetches the actual latest 15-minute global event CSV from GDELT."""
    print("[3] Scanning GDELT Live Feed for Global Macro Events...")
    
    try:
        # 1. Get the URL for the latest 15-minute update
        update_url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
        response = requests.get(update_url)
        export_zip_url = response.text.split('\n')[0].split(' ')[2]
        
        print(f"    -> Downloading latest batch: {export_zip_url.split('/')[-1]}")
        
        # 2. Download and unzip in memory
        r = requests.get(export_zip_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_filename = z.namelist()[0]
        
        # 3. Read into Pandas
        columns_to_keep = {
            27: 'EventCode',            # The CAMEO code
            31: 'GoldsteinScale',       # -10 to +10 severity
            34: 'NumArticles',          # How viral is this event?
            52: 'ActionGeo_CountryCode',# Where did it happen?
            60: 'SourceURL'             # Link to the news article
        }
        
        df = pd.read_csv(
            z.open(csv_filename), 
            sep='\t', 
            header=None, 
            usecols=columns_to_keep.keys(),
            dtype={27: str} 
        )
        df.rename(columns=columns_to_keep, inplace=True)
        
        # 4. Filter for our target country and significant media coverage
        filtered_df = df[(df['ActionGeo_CountryCode'] == target_country) & (df['NumArticles'] > 5)]
        
        if filtered_df.empty:
            print("    -> No high-impact events found. Falling back to global check...")
            filtered_df = df[df['NumArticles'] > 20] # Grab biggest global event instead
            
        top_event = filtered_df.sort_values(by='NumArticles', ascending=False).iloc[0]
        
        # 5. Map the raw data
        raw_cameo = str(top_event['EventCode']).zfill(2)
        root_cameo = raw_cameo[:2]
        event_readable = CAMEO_ROOT_MAP.get(root_cameo, f"Unknown Event ({raw_cameo})")
        
        vuln_trigger = "GENERAL_MACRO"
        if root_cameo in ["16", "17", "19", "20"]:
            vuln_trigger = "GEOPOLITICAL_RISK"
        elif root_cameo in ["04", "06", "07"]:
            vuln_trigger = "FOREIGN_INSTITUTIONAL_FLOWS"
            
        final_event = {
            "event_type": event_readable,
            "location": top_event['ActionGeo_CountryCode'],
            "goldstein_score": float(top_event['GoldsteinScale']),
            "vulnerability_trigger": vuln_trigger,
            "source_url": top_event['SourceURL']
        }
        
        print(f"    -> DETECTED: {final_event['event_type']} in {final_event['location']} (Severity: {final_event['goldstein_score']})")
        return final_event
        
    except Exception as e:
        print(f"    -> Error fetching GDELT data: {e}")
        return {"event_type": "Error fetching data", "goldstein_score": 0}

# ==========================================
# STEP 4: THE MICRO ENGINE (NewsData.io)
# ==========================================
def fetch_company_news(company_name):
    """Fetches the latest articles for a specific company."""
    print(f"[4] Fetching latest news for {company_name}...")
    
    if not NEWSDATA_API_KEY:
        print("    -> No API Key found. Using simulated news data.")
        return [
            f"{company_name} announces record EV sales in domestic market.",
            f"Supply chain disruptions threaten {company_name}'s Q3 delivery targets."
        ]
        
    try:
        url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={company_name}&language=en"
        response = requests.get(url).json()
        articles = [article['title'] for article in response.get('results', [])[:5]]
        print(f"    -> Found {len(articles)} live articles.")
        return articles
    except Exception as e:
        print(f"    -> Error fetching news: {e}")
        return []

# ==========================================
# STEP 5: AI PROCESSING (Groq & Gemini)
# ==========================================
def tag_sentiment_with_groq(articles):
    """Uses Groq (Fast/Free) to tag sentiment and extract keywords."""
    print("[5] Tagging article sentiment with Groq Llama 3.1...")
    
    if not GROQ_API_KEY or not AI_LIBRARIES_INSTALLED:
        print("    -> No Groq API Key found. Using simulated sentiment tags.")
        return {"sentiment": "Bullish", "score": 75, "key_trend": "Strong domestic sales"}
        
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        Read the following articles and return a raw JSON object (no markdown, no extra text) 
        with exactly three keys: 'sentiment' (Bullish, Bearish, or Neutral), 'score' (0-100), 
        and 'key_trend' (a 3-word summary of the trend).
        Articles: {articles}
        """
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"    -> Groq Error: {e}")
        return {"sentiment": "Neutral", "score": 50, "key_trend": "Data unclear"}

def synthesize_with_gemini(macro_event, micro_sentiment, company):
    """Uses Gemini (Reasoning) to write the final Blast Radius explanation."""
    print(f"[6] Writing final Sovereign Synthesis for {company} using Gemini Flash...")
    
    if not GEMINI_API_KEY or not AI_LIBRARIES_INSTALLED:
        print("    -> No Gemini API Key found. Using simulated synthesis text.")
        return f"Despite {micro_sentiment['key_trend']} driving {micro_sentiment['sentiment']} momentum, {company} faces macro headwinds due to a {macro_event['event_type']} event in {macro_event['location']}. With a Goldstein severity score of {macro_event['goldstein_score']}, investors should monitor this conflicting exposure closely."

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are a financial analyst. 
        Company: {company}
        Macro Event Data: {macro_event}
        Company Micro Data: {micro_sentiment}
        Write a concise, 3-sentence 'Sovereign Synthesis' explaining the conflicting exposure 
        between the global macro event and the company's micro sentiment. Be professional and analytical.
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"    -> Gemini Error: {e}")
        return "Error generating synthesis."

# ==========================================
# MAIN WORKER PIPELINE
# ==========================================
def run_pipeline():
    print(f"\n{'='*50}")
    print(f"--- STARTING MUDRA AI PIPELINE AT {datetime.now().strftime('%H:%M:%S')} ---")
    print(f"{'='*50}\n")
    
    # Target Setup
    target_stock = "TATAMOTORS.NS"
    company_name = "Tata Motors"
    
    # 1. Load Rules & Live Data
    sectors = load_sector_map()
    price = get_live_price(target_stock)
    
    # 2. Get Structured Data
    macro_event = fetch_gdelt_events("IN") # IN = India
    micro_news = fetch_company_news(company_name)
    
    # 3. Process with AI
    groq_tags = tag_sentiment_with_groq(micro_news)
    final_synthesis = synthesize_with_gemini(macro_event, groq_tags, company_name)
    
    # 4. Final Database Entry
    final_database_entry = {
        "ticker": target_stock,
        "price": f"₹{price}" if price else "N/A",
        "sentiment": groq_tags['sentiment'],
        "sentiment_score": groq_tags['score'],
        "synthesis": final_synthesis,
    }
    
    print("\n" + "="*50)
    print("--- PIPELINE COMPLETE. SENDING TO SUPABASE: ---")
    print(json.dumps(final_database_entry, indent=2))
    
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            # We use "upsert" so if TATAMOTORS.NS already exists, it just updates it instead of creating duplicates.
            supabase.table("analyses").upsert(final_database_entry, on_conflict="ticker").execute()
            print("    -> SUCCESS: Data securely saved to Supabase Cloud Memory.")
        except Exception as e:
            print(f"    -> SUPABASE ERROR: {e}")
    else:
        print("    -> SKIPPED SUPABASE: Keys not found in environment.")

    print("="*50 + "\n")

# Run the script
if __name__ == "__main__":
    run_pipeline()