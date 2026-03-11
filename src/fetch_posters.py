import os
import asyncio
import aiohttp
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

API_KEY = os.environ["TMDB_API_KEY"]
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/original"

OUTPUT_DIR = ROOT / "requested_posters"

MAX_RETRIES = 3
CONCURRENT_CONNECTIONS = 20


async def _fetch(session, url, params=None, return_json=True):
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    await asyncio.sleep(int(resp.headers.get("Retry-After", 1)))
                    continue
                resp.raise_for_status()
                return await resp.json() if return_json else await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Request failed: {url} | Error: {e}")
            await asyncio.sleep(2 ** attempt)
    return None


async def search_movie(session, query):
    """Search TMDB for a movie by title, return the first result."""
    params = {"api_key": API_KEY, "query": query}
    data = await _fetch(session, f"{BASE_URL}/search/movie", params=params)
    if data and data.get("results"):
        return data["results"][0]
    return None


async def get_variant_posters(session, movie_id):
    """Fetch all poster variants for a given movie id."""
    params = {"api_key": API_KEY, "include_image_language": "en"}
    data = await _fetch(session, f"{BASE_URL}/movie/{movie_id}/images", params=params)
    if data and data.get("posters"):
        return data["posters"]
    return []


async def download_poster(session, semaphore, url, dest):
    """Download a single poster image."""
    async with semaphore:
        data = await _fetch(session, url, return_json=False)
        if data:
            dest.write_bytes(data)
            return True
    return False


async def fetch_all_posters(movie_name, clear_posters=True):
    """Search for a movie and download all variant posters into requested_posters/<movie>/."""
    if clear_posters and OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("*.jpg"):
            f.unlink()
        print(f"Cleared {OUTPUT_DIR}/")

    async with aiohttp.ClientSession() as session:
        movie = await search_movie(session, movie_name)
        if not movie:
            print(f"Movie not found: {movie_name}")
            return

        movie_id = movie["id"]
        title = movie["title"]
        print(f"Found: {title} (id={movie_id})")

        posters = await get_variant_posters(session, movie_id)
        if not posters:
            print(f"No posters found for {title}")
            return

        # Sanitize folder name
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()
        movie_dir = OUTPUT_DIR / safe_title
        movie_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(CONCURRENT_CONNECTIONS)
        tasks = []
        for i, poster in enumerate(posters):
            file_path = poster["file_path"]
            lang = poster.get("iso_639_1") or "noLang"
            dest = movie_dir / f"{i+1:03d}_{lang}{Path(file_path).suffix}"
            url = f"{IMAGE_BASE}{file_path}"
            tasks.append(download_poster(session, semaphore, url, dest))

        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading"):
            results.append(await coro)

        downloaded = sum(1 for r in results if r)
        print(f"Downloaded {downloaded}/{len(posters)} posters to {movie_dir}/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fetch_posters.py <movie name>")
        sys.exit(1)

    movie_name = " ".join(sys.argv[1:])
    asyncio.run(fetch_all_posters(movie_name))
