
# README

Template research project for testing and development

## Features

*   **Custom Logging**: Includes a `LightLogger` class (a wrapper around Python's `logging` module) and a `loguru` setup for structured, colorful, and file-based logging.
*   **Email Notifications**: The `loguru` setup is configured to send email notifications for error-level logs using Apprise.
*   **Environment Configuration**: Uses `.env` files to manage sensitive information like email credentials.

## Project Structure

```
templater-research/
├── .env                  # Environment variables (e.g., API keys, credentials)
├── .gitignore            # Specifies intentionally untracked files that Git should ignore
├── .python-version       # Specifies Python version for pyenv
├── LICENSE               # Project license file
├── README.md             # This file
├── logs/                 # Directory for log files
├── pyproject.toml        # Python project configuration
├── requirements.txt      # Project dependencies
└── src/                  # Source code directory
    ├── __init__.py       # Makes src a Python package
    ├── config.py         # Configuration file (currently empty)
    ├── data.py           # Data loading and processing (currently empty)
    ├── eval.py           # Evaluation scripts (currently empty)
    ├── logger.py         # Custom logging setup
    ├── loss.py           # Loss functions (currently empty)
    ├── main.py           # Main application script (currently empty)
    ├── trainer.py        # Training scripts (currently empty)
    └── utils.py          # Utility functions (currently empty)
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd templater-research
    ```

2.  **Set up a Python virtual environment:**

3.  **Install dependencies:**

4.  **Set up environment variables:**

## Usage

## Dependencies

The main dependencies are listed in `pyproject.toml`:

*   **apprise**: For sending notifications (used for email alerts on errors).
*   **python-dotenv**: For loading environment variables from a `.env` file.
*   **loguru**: A delightful logging library for Python.

Python version required: `>=3.12`
