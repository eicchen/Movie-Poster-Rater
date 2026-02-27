import json
import asyncio
import aiohttp
from pathlib import Path
from api import fetch_genre_list, scrape, keyword_map, clear_posters

async def data_download(clear_poster = False):
    if clear_poster:
        clear_posters()

    async with aiohttp.ClientSession() as session:
        await fetch_genre_list(session)

    df = await scrape()
    
    df.to_parquet("movies.parquet", index=False)
    print(f"Saved {len(df)} movies to movies.parquet")

    if keyword_map:
        Path("keywords.json").write_text(json.dumps(keyword_map))
        print(f"Saved {len(keyword_map)} keywords to keywords.json")

if __name__ == "__main__":
    asyncio.run(data_download(clear_poster = True))