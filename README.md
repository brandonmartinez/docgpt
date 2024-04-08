# docgpt

A sample ChatGPT-style engine that interacts with documents stored in a vector
database.

## Prerequisites

- Python 3.7 or later
- [Ollama](https://ollama.com/)
  - Ollama mistral image (`ollama pull mistral`)
  - Ollama needs to be running on the local host if using a Dev Container and
    [exposed for remote access](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server)
- Setup a virtual environment and source it (if you already have an environment,
  just source it): `python3 -m venv .venv && source .venv/bin/activate`
- Install pip dependencies: `pip install -r requirements.txt`
  - May require `cmake` depending on OS (e.g. `apt-get install cmake` on Ubuntu,
    `brew install cmake` on macOS)

## TODOS:

- look into https://pipenv.pypa.io/en/latest/

## Credits

Based off the work done by [vndee](https://github.com/vndee/local-rag-example).
