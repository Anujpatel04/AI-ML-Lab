# Meeting Notes to Action Agent

A Streamlit application that transforms raw meeting transcripts into structured, actionable outputs using OpenAI.

## Features

- **Meeting Summary**: Generates concise, executive-level summaries
- **Action Items Extraction**: Identifies tasks with assigned owners and deadlines
- **Follow-up Email Generation**: Creates professional, ready-to-send emails

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository and navigate to the project directory:

```bash
cd AI_AGENTS/Meeting_Summarize
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your environment variables by adding your OpenAI API key to the `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the application:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

### Input

Paste your meeting transcript or notes into the text area and click "Generate Actions".

### Output

The application produces three outputs:

1. **Summary**: A brief overview of the meeting discussion
2. **Action Items**: A table containing:
   - Task description
   - Owner (person responsible)
   - Deadline (or "Not specified" if not mentioned)
3. **Follow-up Email**: A professional email ready to be sent to attendees

## Example

**Input:**
```
We discussed finalizing the API design. John will complete the backend by Friday.
Sarah will review the UI next week. Deployment timeline is still undecided.
```

**Output:**
- Summary of key discussion points
- Two action items with owners and deadlines
- Professional follow-up email referencing the meeting

## Configuration

The application uses the OpenAI GPT-4o-mini model by default. The API key is loaded from the `.env` file located two directories above the application.

## License

MIT License
