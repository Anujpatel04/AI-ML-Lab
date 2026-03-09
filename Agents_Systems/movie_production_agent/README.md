# AI Movie Production Agent

A Streamlit application that helps you develop movie concepts by automating script writing and casting suggestions. The agent uses a team of AI agents working together to create compelling movie outlines and suggest suitable actors.

## Features

- Generates script outlines with character descriptions and key plot points
- Suggests actors for main roles based on past performances
- Provides a complete movie concept overview combining script and casting

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Anujpatel04/Anuj-AI-ML-Lab.git
cd Single_AI_Agents/ai_movie_production_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
   - Add your DeepSeek API key to the root `.env` file as `DEEPSEEK_API_KEY`
   - The SerpAPI key is already configured in the code

4. Run the application:
```bash
streamlit run movie_production_agent.py
```

## How It Works

The application uses a team of three specialized agents:

- **ScriptWriter**: Creates script outlines with character descriptions and plot structure based on your movie idea, genre, and target audience
- **CastingDirector**: Suggests actors for main roles by searching for current availability and past performances
- **MovieProducer**: Coordinates the workflow between agents and provides a final summary of the complete movie concept

## Repository

Part of the [Anuj AI/ML Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab) collection.
