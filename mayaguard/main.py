#!/usr/bin/env python
"""
main.py — MayaGuard development entry point.

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
        reload=True,
    )


def dashboard() -> None:
    subprocess.run(
        ["streamlit", "run", "frontend/dashboard.py"],
        check=True,
    )


def test() -> None:
    subprocess.run(["pytest", "tests/", "-v", "--tb=short"], check=True)


def pull_model() -> None:
    from core.config import get_settings
    model = get_settings().ollama_model
    print(f"Pulling model: {model}")
    subprocess.run(["ollama", "pull", model], check=True)


COMMANDS = {
    "serve": serve,
    "dashboard": dashboard,
    "test": test,
    "pull-model": pull_model,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}. Options: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd]()
