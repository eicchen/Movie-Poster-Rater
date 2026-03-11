import json
import asyncio
import aiohttp
from pathlib import Path
from .api import fetch_genre_list, scrape, keyword_map, clear_posters

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

async def data_download(clear_poster = False):
    if clear_poster:
        clear_posters()

    async with aiohttp.ClientSession() as session:
        await fetch_genre_list(session)

    df = await scrape()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / "movies.parquet", index=False)
    print(f"Saved {len(df)} movies to {DATA_DIR / 'movies.parquet'}")

    if keyword_map:
        (DATA_DIR / "keywords.json").write_text(json.dumps(keyword_map))
        print(f"Saved {len(keyword_map)} keywords to {DATA_DIR / 'keywords.json'}")

if __name__ == "__main__":
    asyncio.run(data_download(clear_poster = True))
