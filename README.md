# Argentinian Spanish Translator

An interactive application that translates English or Spanish text into authentic Argentinian Spanish with regional slang, colloquialisms, and expressions.

## 🌟 Key Features

- **Natural Language Translation**: Converts standard English/Spanish to authentic Argentinian Spanish
- **Context-Aware Translations**: Uses a curated database of Argentinian expressions and slang
- **Voice Mode Support**: Speak and get translations through voice using OpenAI's Realtime API
- **Interactive Web UI**: Clean interface powered by Chainlit
- **Vector Search**: Semantic search for finding relevant Argentinian expressions

## 🛠️ Technology Stack

- **LangChain**: For managing LLM interactions and prompt templates
- **OpenAI**: For high-quality translations and voice processing
- **FAISS**: For efficient vector similarity search
- **Chainlit**: For the interactive web interface
- **PDM**: Python package management

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key
- PDM (Python package manager)

### Installation

1. Clone the repository
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

- `app.py`: Main application with Chainlit UI and handlers
- `translator.py`: Core translation logic using LangChain
- `loaders/csv_loader.py`: Loads and processes Argentinian Spanish data
- `realtime/`: Handles OpenAI's Realtime API for voice interactions
- `phrases.csv`: Database of Argentinian expressions

## 🤝 Contributing

Contributions are welcome! Feel free to add more Argentinian expressions to the CSV file or improve the translation logic.

## 📝 License

MIT
