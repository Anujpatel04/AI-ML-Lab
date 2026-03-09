# LinkedIn Profile Roster Agent

An AI-powered tool that analyzes LinkedIn profile screenshots and provides detailed feedback with visual annotations. Get personalized suggestions to improve your LinkedIn profile through intelligent analysis and constructive roasts.

## Features

- **AI-Powered Analysis**: Uses Google Gemini Vision API to analyze profile content
- **Visual Annotations**: Handwritten-style feedback directly on your profile screenshot
- **Smart Suggestions**: Generates roasts, improvements, and roster recommendations
- **Web Interface**: Easy-to-use web app with drag-and-drop upload
- **Color-Coded Feedback**: Red for roasts, green for improvements, blue/orange for roster

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (optional - default key is configured)
export GEMINI_API_KEY="your-api-key-here"
```

### Usage

**Web Interface (Recommended)**
```bash
python app.py
# Open http://localhost:5001 in your browser
```

**Command Line**
```bash
python linkedin_roster_agent.py path/to/profile_screenshot.jpg
python linkedin_roster_agent.py input.jpg -o output.jpg
```

## How It Works

1. Upload a LinkedIn profile screenshot
2. AI analyzes the profile and extracts key information
3. Generates roasts, improvement suggestions, and roster recommendations
4. Creates an annotated image with handwritten-style feedback and arrows pointing to sections

## Output

The tool generates an annotated screenshot with:
- **Roasts**: Humorous but constructive feedback on areas needing improvement
- **Improvements**: Specific, actionable suggestions for each section
- **Roster**: Recommendations for professional categories your profile fits

All feedback is color-coded and positioned near relevant sections with arrows pointing to specific areas.

## Requirements

- Python 3.8+
- Google Gemini API key (free tier: 20 requests/day)
- LinkedIn profile screenshot (PNG, JPG, JPEG, GIF, WEBP)

## Troubleshooting

**API quota exceeded**: You've hit the daily limit. Wait until tomorrow or upgrade your plan.

**Port in use**: Change the port with `PORT=8080 python app.py`

**Poor results**: Use high-resolution screenshots that include all profile sections

## License

MIT License - feel free to use and modify as needed.
