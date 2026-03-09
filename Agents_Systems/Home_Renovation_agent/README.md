# AI Home Renovation Planner Agent

A multi-agent system built with Google ADK that analyzes room photos, creates personalized renovation plans, and generates photorealistic renderings using Gemini 2.5 Flash's multimodal capabilities.

## Overview

This agent helps you plan home renovations by analyzing your space, understanding your budget constraints, and generating detailed renovation plans with visual renderings. It uses a coordinator/dispatcher pattern with specialized agents for different aspects of renovation planning.

## Features

- **Image Analysis**: Upload room photos and inspiration images for automatic analysis
- **Photorealistic Rendering**: Generate professional-quality images of renovated spaces
- **Budget Planning**: Get recommendations tailored to your budget constraints
- **Complete Roadmaps**: Receive timeline, budget breakdown, and action checklists
- **Iterative Refinement**: Edit and refine renderings based on feedback

## Architecture

The system uses a Coordinator/Dispatcher pattern with three specialized agents:

1. **Visual Assessor**: Analyzes uploaded photos, extracts style from inspiration images, estimates costs, and identifies improvement opportunities

2. **Design Planner**: Creates budget-appropriate design plans, specifies materials and colors, and prioritizes high-impact changes

3. **Project Coordinator**: Generates comprehensive renovation roadmaps, creates photorealistic renderings, and provides budget breakdowns and timelines

## Prerequisites

- Python 3.10 or higher
- Google API key (Gemini API)
- Google ADK installed

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your Google API key:
   
   **Important**: This agent loads the API key from the root `.env` file of the repository. No API keys are stored in this folder.
   
   Add your Google API key to the `.env` file in the root directory (`/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env`):
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   
   Alternatively, you can export it as an environment variable:
   ```bash
   export GOOGLE_API_KEY="your_google_api_key_here"
   ```

## Running the Agent

1. Navigate to the parent directory containing all agents:
   ```bash
   cd /path/to/Single_AI_Agents
   ```

2. Launch ADK Web:
   ```bash
   adk web
   ```

3. Open your browser and navigate to the URL shown (typically `http://localhost:8000`)

4. Select `Home_Renovation_agent` from the list of available agents

## Usage Examples

### Basic Renovation Planning
Upload a photo of your room and ask:
```
"What can I improve here with a $5k budget?"
```

### Style Transformation
Upload your current room photo and an inspiration image:
```
"Transform my kitchen to look like this. What's the cost?"
```

### Text-Based Planning
Describe your space:
```
"Renovate my 10x12 kitchen with oak cabinets and laminate counters. 
Want modern farmhouse style with white shaker cabinets. Budget: $30k"
```

### Iterative Refinement
After generating an initial rendering:
```
"Make the cabinets cream instead of white"
"Add pendant lights over the island"
"Change flooring to lighter oak"
```

## Sample Prompts

- "I want to renovate my small galley kitchen. It's 8x12 feet, has oak cabinets from the 90s. I love modern farmhouse style. Budget: $25k"
- "My master bathroom is tiny (5x8) with a cramped tub. I want a spa-like retreat with walk-in shower. Budget: $15k"
- "Transform my boring bedroom into a cozy retreat. Thinking accent wall, new flooring. Budget: $12k"

## Capabilities

- **Google Search**: Finds renovation costs, materials, and design trends
- **Cost Estimation**: Calculates costs by room type and renovation scope
- **Timeline Calculation**: Estimates project duration
- **Rendering Generation**: Creates photorealistic renderings of renovated spaces
- **Rendering Editing**: Refines renderings based on user feedback
- **Version Control**: Automatic version tracking for all generated renderings

## Technical Details

This agent demonstrates a Coordinator/Dispatcher pattern combined with a Sequential Pipeline:

```
Coordinator (Root Agent)
    ├── Info Agent (quick Q&A)
    └── Planning Pipeline (Sequential)
          ├── Visual Assessor (image analysis)
          ├── Design Planner (specifications)
          └── Project Coordinator (rendering + roadmap)
```

This architecture provides:
- Efficient workflow execution (only runs what's needed)
- Modular design with clear agent responsibilities
- Scalability for adding new features
- Production-ready agentic system pattern

## Repository

This agent is part of the Anuj AI/ML Lab repository:
- GitHub: [https://github.com/Anujpatel04/Anuj-AI-ML-Lab/tree/main/Single_AI_Agents/Home_Renovation_agent](https://github.com/Anujpatel04/Anuj-AI-ML-Lab/tree/main/Single_AI_Agents/Home_Renovation_agent)

## Requirements

- google-adk>=1.15.0
- google-generativeai>=0.8.3
- python-dotenv>=1.0.0

## Notes

- **API Key Configuration**: The agent automatically loads the `GOOGLE_API_KEY` from the root `.env` file. No API keys are hardcoded in this folder.
- The agent requires a valid Google API key with access to Gemini models
- Image analysis capabilities depend on Gemini 2.5 Flash's multimodal features
- Rendering generation may take some time depending on complexity
- All generated renderings are automatically versioned for easy reference
