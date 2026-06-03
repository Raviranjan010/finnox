import feedparser
import httpx
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class Article:
    text: str
    source: str
    published: str
    ticker_mentions: List[str]

class NewsIngestionPipeline:
    """News Ingestion Pipeline — real-time multi-source ingestion (RSS, SEC EDGAR)"""
    
    SOURCES = {
        "reuters":    "https://feeds.reuters.com/reuters/businessNews",
        "bloomberg":  "https://feeds.bloomberg.com/markets/news.rss",
        "sec_edgar":  "https://efts.sec.gov/LATEST/search-index?q=%228-K%22&dateRange=custom",
        "ft":         "https://www.ft.com/rss/home/uk",
    }
    
    async def ingest(self, tickers: List[str]) -> List[Article]:
        articles = []
        headers = {
            "User-Agent": "FinnoxAnalytics contact@finnoxanalytics.com"
        }
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            for source, url in self.SOURCES.items():
                try:
                    if source == "sec_edgar":
                        # SEC EDGAR EFTS is a JSON Elasticsearch endpoint
                        response = await client.get(url, timeout=10.0)
                        if response.status_code == 200:
                            data = response.json()
                            hits = data.get("hits", {}).get("hits", [])
                            for hit in hits:
                                source_data = hit.get("_source", {})
                                form = source_data.get("form", "Filing")
                                display_names = source_data.get("display_names", [])
                                display_name = " ".join(display_names)
                                date_str = source_data.get("file_date", "")
                                
                                # Filter by tickers if specified, else include all
                                if not tickers or any(t.lower() in display_name.lower() for t in tickers):
                                    articles.append(Article(
                                        text=f"SEC EDGAR filing {form} for {display_name}",
                                        source=source,
                                        published=date_str,
                                        ticker_mentions=[t for t in tickers if t.lower() in display_name.lower()] if tickers else display_names
                                    ))
                    else:
                        # Standard RSS feed
                        response = await client.get(url, timeout=10.0)
                        if response.status_code == 200:
                            feed = feedparser.parse(response.text)
                            for entry in feed.entries:
                                title = entry.get("title", "")
                                summary = entry.get("summary", title)
                                published = entry.get("published", entry.get("updated", ""))
                                
                                # Filter by tickers if specified, else include all
                                if not tickers or any(t.lower() in title.lower() or t.lower() in summary.lower() for t in tickers):
                                    articles.append(Article(
                                        text=summary,
                                        source=source,
                                        published=published,
                                        ticker_mentions=[t for t in tickers if t.lower() in title.lower() or t.lower() in summary.lower()] if tickers else []
                                    ))
                except Exception as e:
                    logger.warning(f"Failed to ingest from source {source}: {e}")
        return articles
