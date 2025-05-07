import requests
import pandas as pd

def search_scopus(query, api_key):
    """Cari artikel di Scopus berdasarkan query."""
    url = f'https://api.elsevier.com/content/search/scopus?query={query}&apiKey={api_key}&count=25'
    headers = {
        'Accept': 'application/json',
        'X-ELS-APIKey': api_key
    }

    response = requests.get(url, headers=headers)
    results = []

    if response.status_code == 200:
        data = response.json()

        if 'search-results' in data:
            for entry in data['search-results']['entry']:
                title = entry.get('dc:title', 'No title found')
                authors = entry.get('dc:creator', 'No authors found')
                publication_year = entry.get('prism:coverDate', 'No publication date found')
                doi = entry.get('prism:doi', 'No DOI found')
                source_title = entry.get('prism:publicationName', 'No source title found')
                citation_count = entry.get('citedby-count', 'No citation count found')
                issn = entry.get('prism:issn', 'No ISSN found')

                result = {
                    "Title": title,
                    "Authors": authors,
                    "Publication Year": publication_year,
                    "DOI": doi,
                    "Publication Name": source_title,
                    "ISSN": issn,
                    "Citations": citation_count
                }
                results.append(result)
        else:
            print("No results found.")
    else:
        print(f"Request failed with status code: {response.status_code}")

    return results


def get_journal_metrics_by_issn(issn, api_key):
    """Ambil CiteScore, SJR, dan SNIP dari ISSN jurnal."""
    url = f"https://api.elsevier.com/content/serial/title?issn={issn}&apiKey={api_key}"
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": api_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        try:
            data = response.json()
            entry = data['serial-metadata-response']['entry'][0]
            citescore = entry.get('citeScoreYearInfoList', {}).get('citeScoreCurrentMetric', 'N/A')
            sjr = entry.get('SJRList', {}).get('SJR', [{}])[0].get('$', 'N/A')
            snip = entry.get('SNIPList', {}).get('SNIP', [{}])[0].get('$', 'N/A')
            return citescore, sjr, snip
        except Exception as e:
            print(f"Parsing error for ISSN {issn}: {e}")
    else:
        print(f"Metric fetch failed for ISSN {issn}: {response.status_code}")
    return 'N/A', 'N/A', 'N/A'


def enrich_with_metrics(results, api_key):
    """Menambahkan CiteScore, SJR, dan SNIP ke list hasil pencarian."""
    enriched_results = []

    for item in results:
        issn = item.get('ISSN')
        if issn and issn != 'No ISSN found':
            citescore, sjr, snip = get_journal_metrics_by_issn(issn, api_key)
        else:
            citescore, sjr, snip = 'N/A', 'N/A', 'N/A'

        item['CiteScore'] = citescore
        item['SJR'] = sjr
        item['SNIP'] = snip

        enriched_results.append(item)

    return enriched_results
