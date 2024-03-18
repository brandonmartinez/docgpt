# docgpt

A sample ChatGPT-style engine that interacts with documents stored in a vector
database.

## Prerequisites

- Python 3.7 or later
- [Ollama](https://ollama.com/)
- Ollama mistral image (`ollama pull mistral`)
- Setup a virtual environment and source it (if you already have an environment,
  just source it): `python3 -m venv .venv && source .venv/bin/activate`
- Install pip dependencies: `pip install -r requirements.txt`
  - May require `cmake` depending on OS (e.g. `apt-get install cmake` on Ubuntu,
    `brew install cmake` on macOS)

## Credits

Based off the work done by [vndee](https://github.com/vndee/local-rag-example).
