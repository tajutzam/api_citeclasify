import requests
from dotenv import load_dotenv
import os

def get_journal_metrics(issn, api_key):
    url = f"https://api.elsevier.com/content/serial/title?issn={issn}&apiKey={api_key}"

    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": api_key
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        try:
            entry = data['serial-metadata-response']['entry'][0]
            source_title = entry.get('dc:title', 'Unknown')
            citescore = entry.get('citeScoreYearInfoList', {}).get('citeScoreCurrentMetric', 'N/A')
            sjr = entry.get('SJRList', {}).get('SJR', [{}])[0].get('$', 'N/A')
            snip = entry.get('SNIPList', {}).get('SNIP', [{}])[0].get('$', 'N/A')

            return {
                "Journal": source_title,
                "CiteScore": citescore,
                "SJR": sjr,
                "SNIP": snip
            }
        except Exception as e:
            print("Data format error:", e)
            return {}
    else:
        print("Failed to fetch data:", response.status_code)
        return {}

load_dotenv()

api_key = os.getenv('TOKEN')
issn = "0353-4529" 
print(get_journal_metrics(issn, api_key))
