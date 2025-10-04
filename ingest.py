import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

INPUT_FILE = "SB_publication_PMC.csv"   # source metadata
OUTPUT_FILE = "publications_with_text.csv"

def fetch_pmc_text(url: str) -> str:
    """Fetch and clean text from a PMC article page."""
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.content, "html.parser")

        # Try multiple selectors (abstract, body, paragraphs)
        parts = []

        # Abstract
        for div in soup.select("div.abstr, div.abstract"):
            parts.append(div.get_text(" ", strip=True))

        # Main body
        for div in soup.select("div.body, div#maincontent"):
            parts.append(div.get_text(" ", strip=True))

        # Fallback to paragraphs
        if not parts:
            paragraphs = soup.find_all("p")
            parts = [p.get_text(" ", strip=True) for p in paragraphs]

        text = " ".join(parts)
        return " ".join(text.split())  # normalize whitespace

    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}")
        return ""

def main():
    df = pd.read_csv(INPUT_FILE)

    titles, links, texts = [], [], []

    print("\nFetching articles...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row.get("Link", "")
        title = row.get("Title", "Untitled")

        text = fetch_pmc_text(url)
        if len(text) > 500:  # keep only real content
            print(f"✅ Extracted {len(text)} chars from {url}")
            titles.append(title)
            links.append(url)
            texts.append(text)
        else:
            print(f"⚠️ No usable text from {url}")

    # Build new dataframe only with good rows
    clean_df = pd.DataFrame({"Title": titles, "Link": links, "Text": texts})
    clean_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved {len(clean_df)} rows with cleaned text to {OUTPUT_FILE}")
    print(clean_df.head())

if __name__ == "__main__":
    main()


