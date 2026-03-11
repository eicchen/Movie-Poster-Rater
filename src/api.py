import os
import json
import time
import asyncio
import aiohttp
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

API_KEY = os.environ["TMDB_API_KEY"]
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w154"

#lowest resolution
# IMAGE_BASE = "https://image.tmdb.org/t/p/w92"
POSTER_DIR = ROOT / "posters"
POSTER_DIR.mkdir(exist_ok=True)

MAX_RETRIES = 3
CONCURRENT_CONNECTIONS = 50
RATE_LIMIT_REQS = 40
RATE_LIMIT_WINDOW = 1.0

keyword_map = {}
request_timestamps = []
rate_limit_lock = asyncio.Lock()
semaphore = asyncio.Semaphore(CONCURRENT_CONNECTIONS)

async def enforce_rate_limit():
    global request_timestamps
    while True:
        async with rate_limit_lock:
            now = time.time()
            request_timestamps = [t for t in request_timestamps if now - t < RATE_LIMIT_WINDOW]
            if len(request_timestamps) < RATE_LIMIT_REQS:
                request_timestamps.append(now)
                return
            wait = RATE_LIMIT_WINDOW - (now - request_timestamps[0])
        if wait > 0:
            await asyncio.sleep(wait)

async def _fetch(session, url, params=None, return_json=True):
    for attempt in range(MAX_RETRIES):
        await enforce_rate_limit()
        try:
            async with semaphore, session.get(url, params=params) as resp:
                if resp.status == 429:
                    await asyncio.sleep(int(resp.headers.get("Retry-After", 1)))
                    continue
                resp.raise_for_status()
                return await resp.json() if return_json else await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == MAX_RETRIES - 1:
                tqdm.write(f"Request failed: {url} | Error: {e}")
            await asyncio.sleep(2 ** attempt)
    return None

async def api_request(session, endpoint, params=None):
    params = {**(params or {}), "api_key": API_KEY}
    return await _fetch(session, f"{BASE_URL}{endpoint}", params=params, return_json=True)

async def download_poster(session, movie_id, poster_path):
    if not poster_path:
        return None
    local = POSTER_DIR / f"{movie_id}.jpg"
    if local.exists():
        return str(local)

    data = await _fetch(session, f"{IMAGE_BASE}{poster_path}", return_json=False)
    if data:
        local.write_bytes(data)
        return str(local)
    return None

async def fetch_genre_list(session, path=None):
    path = path or ROOT / "data" / "genres.json"
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = await api_request(session, "/genre/movie/list")
    if data:
        path.write_text(json.dumps(data["genres"]))

async def fetch_keywords(session, movie_id):
    data = await api_request(session, f"/movie/{movie_id}/keywords")
    if not data:
        return []
    keywords = data.get("keywords", [])
    for kw in keywords:
        keyword_map[kw["id"]] = kw["name"]
    return [kw["id"] for kw in keywords]

async def process_movie(session, m, year, keywords):
    poster_task = asyncio.create_task(download_poster(session, m["id"], m.get("poster_path")))
    kw_task = asyncio.create_task(fetch_keywords(session, m["id"])) if keywords else None
    
    return {
        "id": m["id"],
        "title": m["title"],
        "rating": m["vote_average"],
        "vote_count": m["vote_count"],
        "genre_ids": m.get("genre_ids", []),
        "keyword_ids": await kw_task if kw_task else [],
        "poster_local": await poster_task,
        "year": year
    }

async def scrape(years=range(2000, 2025), max_per_year=200, max_movies=None, keywords=True):
    records = []
    pbar = tqdm(total=len(years), desc="Years", unit="yr")

    async with aiohttp.ClientSession() as session:
        for year in years:
            year_count = 0                                          # <--
            for page in range(1, (max_per_year // 20) + 2):         # <-- derive page limit
                data = await api_request(session, "/discover/movie", {
                    "page": page,
                    "sort_by": "popularity.desc",
                    "primary_release_year": year
                })
                if not data or not data.get("results"):
                    break

                tasks = [process_movie(session, m, year, keywords) for m in data["results"]]

                for task in asyncio.as_completed(tasks):
                    record = await task
                    records.append(record)
                    year_count += 1                                 # <--
                    pbar.set_postfix(year=year, movie=record["title"][:20])

                    if max_movies and len(records) >= max_movies:
                        pbar.close()
                        return pd.DataFrame(records)

                if year_count >= max_per_year:                      # <--
                    break                                           # <--
                if page >= data.get("total_pages", 1):
                    break

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def clear_posters():
    for f in POSTER_DIR.glob("*.jpg"):
        f.unlink()