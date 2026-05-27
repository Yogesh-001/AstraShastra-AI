# Contributing to MayaGuard

Thank you for your interest in contributing to **MayaGuard**! We welcome all contributions, including bug reports, feature requests, documentation improvements, and code changes.

By contributing to this repository, you help make AI systems more reliable and trustworthy.

---

## 🛠️ Getting Started

### 1. Prerequisite Installations
Ensure you have the following installed locally:
- Python 3.10 or higher
- Docker & Docker Compose (for running Qdrant and Ollama locally)

### 2. Environment Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yogesh-001/mayaguard.git
   cd mayaguard
   ```

2. **Initialize a Virtual Environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install the dependencies in editable development mode:**
   ```bash
   pip install -e ".[dev,frontend]"
   ```

4. **Copy the environment configuration:**
   ```bash
   cp .env.example .env
   ```

---

## 📐 Coding Standards

To maintain codebase health, we adhere to strict quality checks:

### 1. Code Style and Linting
We use **Ruff** for code formatting and import sorting. Before committing your code, format and check your files:
```bash
ruff format .
ruff check . --fix
```

### 2. Static Typing
We use **MyPy** for strict type verification. Ensure all your parameters and return types are fully annotated:
```bash
mypy .
```

---

## 🧪 Testing Guidelines

Before opening a pull request, you must verify that your changes pass all unit and integration tests.

Run the test suite locally:
```bash
python -m pytest -v
```

* **Unit Tests (`tests/unit/`):** Test pure business logic, calculations, and parsers. Keep these fully isolated and fast.
* **Integration Tests (`tests/integration/`):** Test full multi-stage pipeline runs using simulated mock adapters and client stubs to ensure offline robustness and maximum speed.

---

## 📥 Submitting a Pull Request

1. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-awesome-feature
   ```
2. **Commit your changes** with descriptive commit messages following standard conventions (e.g., `feat: add DevOps relabeling specs`).
3. **Verify** style checks (`ruff check`), type checks (`mypy`), and that the test suite (`pytest`) is perfectly green.
4. **Push the branch** and submit a Pull Request (PR) on GitHub.

We look forward to reviewing your contributions!
