# Argentinian Spanish Translator 🧉

An interactive web application that translates English or Spanish text into authentic Argentinian Spanish, complete with regional slang, colloquialisms, and expressions. Built with Chainlit.

## 🌟 Key Features

- **Natural Language Translation**: Converts standard English/Spanish to authentic Argentinian Spanish
- **Context-Aware Translations**: Uses a curated database of Argentinian expressions and slang
- **Interactive Web UI**: Clean interface powered by Chainlit
- **Vector Search**: Semantic search for finding relevant Argentinian expressions
- **Prompt Management**: Easy-to-edit prompts stored in markdown files

## 🛠️ Technology Stack

- **LangChain**: For managing LLM interactions and prompt templates
- **OpenAI**: For high-quality translations
- **FAISS**: For efficient vector similarity search
- **Chainlit**: For the interactive web interface
- **PDM**: Python package management

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key
- PDM (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/arg-translator.git
   cd arg-translator
   ```

2. Install dependencies:
   ```bash
   pdm install
   ```

3. Create a `.env` file based on `.env.example`:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

```bash
pdm run chainlit run app.py
```

The application will be available at http://localhost:8000

## 📊 Data Structure

The application uses a CSV file (`phrases.csv`) with Argentinian expressions following this format:

- **Original Phrase/Word**: The standard English or Spanish phrase
- **Argentinian Equivalent**: The Argentinian Spanish version
- **Explanation (Context/Usage)**: When and how to use the expression
- **Region Specificity**: Where in Argentina the expression is commonly used
- **Level of Formality**: Formal, casual, slang, etc.

## 🧩 Project Structure

```
arg-translator/
├── app.py                 # Main Chainlit application, UI event handlers
├── config.py              # Configuration settings (API keys, paths)
├── core/
│   ├── data_loader.py     # Loads CSV, creates Documents & vector store
│   ├── prompt_manager.py  # Loads prompt templates from files
│   └── translator.py      # Core LangChain RAG translation logic
├── data/
│   └── phrases.csv        # Database of Argentinian expressions
├── prompts/               # Directory for prompt markdown files
│   ├── system.md          # System prompt for the translator
│   └── translation.md     # Translation prompt template
├── services/
│   └── translation_service.py # Service layer orchestrating translation
├── chainlit.md            # Chainlit welcome screen content
├── .env.example           # Example environment variables
└── README.md              # Project documentation
```

## 🎯 Usage Examples

### Text Translation
```
User: Hello, how are you?
Bot: ¡Hola che! ¿Cómo andás?
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Add more Argentinian expressions to `phrases.csv`
2. Improve translation prompts in `prompts/`
3. Enhance the UI/UX
4. Add new features or bug fixes

## 📝 License

MIT License - feel free to use this project for your own purposes.

## 🙏 Acknowledgments

- Chainlit team for the amazing web interface
- The LangChain community for their excellent tools
