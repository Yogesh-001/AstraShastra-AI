#!/usr/bin/env python
"""
CLI entry point for managing the MayaGuard server, database, and test tools.

Usage:
    # Start the API server
    python main.py serve

    # Start the Streamlit dashboard
    python main.py dashboard

    # Run the unit tests
    python main.py test

    # Pull the default Ollama model
    python main.py pull-model
"""

import subprocess
import sys


def serve() -> None:
    from core.logging import setup_logging
    setup_logging()
    import uvicorn
    from core.config import get_settings
    s = get_settings()
    uvicorn.run(
        "serving.app:app",
        host=s.api_host,
        port=s.api_port,
        workers=s.api_workers,
        reload=False,
    )


def dashboard() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "frontend/dashboard.py"],
        check=True,
    )


def test() -> None:
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], check=True)


def pull_model() -> None:
    from core.config import get_settings
    model = get_settings().ollama_model
    print(f"Pulling model: {model}")
    subprocess.run(["ollama", "pull", model], check=True)


def seed() -> None:
    import asyncio
    asyncio.run(_async_seed())


async def _async_seed() -> None:
    from pathlib import Path
    import json
    from core.models import Document
    from core.retrieval.retriever import Retriever
    from serving.registry import _adapters

    print("[START] Starting MayaGuard database seeding pipeline...")
    
    datasets_dir = Path(__file__).parent / "benchmarks" / "datasets"
    
    for name, adapter in _adapters.items():
        if name == "default":
            seed_file = datasets_dir / "medical_seed.json"
        else:
            seed_file = datasets_dir / f"{name}_seed.json"
            
        if not seed_file.exists():
            print(f"[WARN] Seed file not found for {name}: {seed_file}")
            continue
            
        print(f"[LOAD] Loading dataset for adapter '{name}' from {seed_file.name}...")
        try:
            with open(seed_file, "r") as f:
                raw_data = json.load(f)
            
            docs = [
                Document(
                    content=item["content"],
                    source=item["source"],
                    metadata=item.get("metadata", {})
                )
                for item in raw_data
            ]
            
            cfg = adapter.get_retriever_config()
            print(f"[INFO] Initializing Qdrant collection '{cfg['collection']}'...")
            retriever = None
            try:
                retriever = await Retriever.create(**cfg)
                print(f"[INFO] Indexing {len(docs)} documents into collection...")
                upserted = await retriever.upsert(docs)
                print(f"[OK] Seeding complete for '{name}': {upserted} documents indexed.")
            finally:
                if retriever is not None:
                    await retriever.close()
        except Exception as exc:
            print(f"[ERROR] Failed to seed adapter '{name}': {exc}")
            
    print("[SUCCESS] Seeding process completed successfully!")


COMMANDS = {
    "serve": serve,
    "dashboard": dashboard,
    "test": test,
    "pull-model": pull_model,
    "seed": seed,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}. Options: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd]()
