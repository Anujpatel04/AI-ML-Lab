# AI Health & Fitness Planner

A personalized health and fitness planning application powered by DeepSeek API. Generates tailored dietary and fitness plans based on user profile information.

## Features

- Personalized dietary plans with meal recommendations
- Customized fitness routines based on goals
- Interactive Q&A about generated plans
- Supports various dietary preferences (Vegetarian, Keto, Low Carb, etc.)

## Requirements

- Python 3.10+
- DeepSeek API key (stored in `.env` file at project root)

## Installation

1. Install dependencies:
   ```bash
    pip install -r requirements.txt
    ```

2. Add your DeepSeek API key to `.env` file in the project root:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```

## Usage

Run the Streamlit application:
    ```bash
    streamlit run health_agent.py
    ```

Enter your profile information (age, weight, height, activity level, dietary preferences, fitness goals) and generate personalized plans.

## Technologies

- Streamlit
- Agno AI Agent Framework
- DeepSeek API
