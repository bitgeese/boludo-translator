# Argentinian Spanish Translator 🧉

An interactive web application that translates English or Spanish text into authentic Argentinian Spanish, complete with regional slang, colloquialisms, and expressions. Built with Chainlit.

## 🌟 Key Features

- **Natural Language Translation**: Converts standard English/Spanish to authentic Argentinian Spanish.
- **Smart Language Detection**: Automatically detects input language (English/Spanish). Uses LLM for short inputs and a faster statistical model for longer inputs.
- **Context-Aware Translations**: Leverages RAG with multiple data sources for authentic results:
  - **Phrase Database**: Curated collection of Argentinian expressions and slang
  - **VentureOut Spanish Content**: Articles and lessons about Rioplatense Spanish
- **Interactive Web UI**: Clean interface powered by Chainlit.
- **Vector Search**: Employs FAISS or Chroma for efficient semantic search of relevant context.
- **Prompt Management**: Centralized and easy-to-edit prompts.
- **Configurable**: Settings managed via `config.py` and `.env`.

## 🛠️ Technology Stack

- **LangChain**: LLM orchestration, RAG implementation, prompt management.
- **OpenAI**: Core LLM for translation and language detection.
- **FAISS/Chroma**: Efficient vector similarity search.
- **Langdetect**: Statistical language detection for longer texts.
- **Chainlit**: Interactive web UI framework.
- **PDM**: Python dependency management.

## 🚀 Getting Started

### Prerequisites

- Python 3.12+ (Update if necessary, check `pyproject.toml`)
- OpenAI API key
- PDM (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/arg-translator.git # Replace with your actual repo URL
   cd arg-translator
   ```

2. Install dependencies using PDM:
   ```bash
   pdm install
   ```

3. Create and configure the `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

```bash
pdm run chainlit run app.py -w # The -w flag enables auto-reload on code changes
```

The application will typically be available at `http://localhost:8000`.

## 📊 Data Sources

The application uses multiple data sources for context-aware translations:

### 1. Phrase Database (CSV)
Located at `data/phrases.csv` containing Argentinian expressions. The expected columns are:

- `Original Phrase/Word`: The standard English or Spanish phrase.
- `Argentinian Equivalent`: The Argentinian Spanish version.
- `Explanation (Context/Usage)`: Guidance on when and how to use the expression.
- `Region Specificity` (Optional): Where in Argentina the expression is commonly used.
- `Level of Formality` (Optional): E.g., formal, casual, slang.

### 2. VentureOut Spanish Content (JSONL)
Located at `data/ventureout_data.jsonl` containing articles and lessons about Rioplatense Spanish from [VentureOut Spanish](https://ventureoutspanish.com/).

- Content is scraped and cleaned using scripts in the `data_scripts/` directory
- Each record contains the URL, title, and cleaned content
- Provides rich context about Argentinian Spanish usage

## 🧩 Project Structure

```plaintext
arg-translator/
├── .chainlit/             # Chainlit configuration and assets
├── .venv/                 # Virtual environment (managed by PDM)
├── core/                  # Core logic (RAG, data loading, prompts)
│   ├── __init__.py
│   ├── data_loader.py     # Loads CSV, JSONL, creates vector store
│   ├── ventureout_loader.py # Handles VentureOut Spanish data
│   ├── exceptions.py      # Custom exception classes
│   ├── prompt_manager.py  # Loads prompt templates from files
│   └── translator.py      # Core LangChain RAG translation logic
├── data/                  # Data files used by the application
│   ├── phrases.csv        # Database of Argentinian expressions
│   └── ventureout_data.jsonl # VentureOut Spanish content
├── data_scripts/          # Scripts for data collection and processing
│   ├── scrape_ventureout.py  # Website scraper for VentureOut
│   └── README.md          # Documentation for data scripts
├── prompts/               # Directory for prompt markdown files
│   ├── system.md          # System prompt for the translator
│   └── translation.md     # Translation prompt template
├── public/                # Static assets for Chainlit UI (if any)
├── services/              # Service layer (business logic)
│   ├── __init__.py
│   └── translation_service.py # Orchestrates detection & translation
├── app.py                 # Main Chainlit application entrypoint
├── chainlit.md            # Chainlit welcome screen content
├── config.py              # Pydantic settings configuration
├── icon.py                # Chainlit icon generation script (optional)
├── pdm.lock               # PDM lock file for reproducible builds
├── pyproject.toml         # Project metadata and dependencies (PDM)
├── README.md              # This file
├── .env                   # Local environment variables (API keys, etc.) - **DO NOT COMMIT**
├── .env.example           # Example environment variables file
└── .gitignore             # Specifies intentionally untracked files
```

## 🎯 Usage Examples

### Text Translation

```
User: Hello, how are you?
Bot: ¡Hola che! ¿Cómo andás?

User: cool
Bot: Piola

User: bonjour
Bot: Sorry, I currently only support English ('en') and Spanish ('es'). Detected: fr
```

## 🤝 Contributing

Contributions are welcome! Please consider:

1.  Adding more Argentinian expressions to `data/phrases.csv`.
2.  Improving translation prompts in `prompts/`.
3.  Refining the language detection logic or prompts.
4.  Enhancing the Chainlit UI/UX.
5.  Adding tests.
6.  Reporting bugs or suggesting features via Issues.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details (if applicable, or state MIT License).

## 🙏 Acknowledgments

- The [Chainlit](https://chainlit.io/) team for the excellent UI framework.
- The [LangChain](https://www.langchain.com/) community for their powerful tools.
- OpenAI for the underlying language models.
- [VentureOut Spanish](https://ventureoutspanish.com/) for their valuable content on Rioplatense Spanish.
