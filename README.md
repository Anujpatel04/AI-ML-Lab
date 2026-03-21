<div align="center">

# Anuj AI/ML Lab

### *Your Gateway to Production-Ready AI Systems*

[![Status](https://img.shields.io/badge/Status-Active%20Development-success?style=for-the-badge)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Anujpatel04/Anuj-AI-ML-Lab?style=for-the-badge&logo=github)](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)
[![Forks](https://img.shields.io/github/forks/Anujpatel04/Anuj-AI-ML-Lab?style=for-the-badge&logo=github)](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=2E9EF7&center=true&vCenter=true&width=800&lines=AI+Agents+%7C+RAG+Systems+%7C+ML+From+Scratch;Production+Ready+%7C+Educational+%7C+Experimental;Voice+Agents+%7C+MCP+Integration+%7C+LLM+Fine-tuning" alt="Typing SVG" />

---

### A comprehensive collection of AI agents, RAG applications, and machine learning algorithms built from the ground up

[Quick Start](#quick-start) • [Project Index](#project-categories) • [Setup](docs/SETUP.md) • [Contributing](#contributing) • [Star Us](#)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Categories](#project-categories)
- [Quick Start](#quick-start)
- [LLM Projects (Submodules)](#llm-projects-submodules)
- [Contributing](#contributing)
- [Code of Conduct](#code-of-conduct)
- [License](#license)

---

## Overview

Welcome to **Anuj AI/ML Lab** - a comprehensive learning resource and development playground for building cutting-edge AI systems. This repository bridges the gap between theory and practice, offering production-ready implementations alongside educational resources.

> **Active Development**: This repository undergoes frequent updates with new features and improvements.
> **Star & Fork** this repo to stay updated and experiment with your own modifications!

---

## Key Features

<table>
<tr>
<td width="50%">

### **AI Agents**
- Content generation pipelines
- Intelligent web scraping
- Meeting transcription systems
- Business intelligence agents
- Multi-modal integrations (Gmail, YouTube, PDF)

</td>
<td width="50%">

### **RAG Applications**
- Advanced retrieval systems
- Document processing pipelines
- Knowledge base management
- Context-aware responses
- Multi-source data integration

</td>
</tr>
<tr>
<td width="50%">

### **ML From Scratch**
- Supervised learning algorithms
- Unsupervised learning methods
- Educational implementations
- Performance benchmarks
- Visualization tools

</td>
<td width="50%">

### **Voice & MCP Agents**
- Voice-powered AI assistants
- Customer support automation
- Model Context Protocol integration
- External tool connectivity
- Real-time interactions

</td>
</tr>
</table>

---

## Project Categories

<details open>
<summary><b>AI Agents</b> - Click to expand</summary>

| Project | Description |
|---------|-------------|
| [OpenAI Video Agent](AI_AGENTS/OpenAI_VideoAgent) | Sora short video generation (15–30s), Streamlit + CLI |
| [Meeting Summarize](AI_AGENTS/Meeting_Summarize) | Meeting transcripts to actionable notes and follow-up emails |
| [Meeting Agent](AI_AGENTS/Meeting_Agent) | Meeting transcription and AI summaries |
| [Personal Context Memory](AI_AGENTS/PersonalContextMemory_agent) | Context-aware agent with persistent memory |
| [Domain-Specific Q&A Chatbot](AI_AGENTS/DomainSpecific_Q&A_Chatbot) | Domain-focused question answering |
| [OpenAI Content Rewriter](AI_AGENTS/OpenAI_ContentRewritter_Agent) | Automated content rewriting with OpenAI |
| [Prompt Optimizer](AI_AGENTS/Prompt_optimizer) | Optimize and refine LLM prompts |
| [Multi-Agent Researcher](AI_AGENTS/multi_agent_researcher) | Multi-agent research on HackerNews and web |
| [Resume & Job Suggestions](AI_AGENTS/ResumeJOB_Suggestions) | Resume analysis and job recommendations |
| [Chat with SQL](AI_AGENTS/ChatWith_SQL_Locally) | Natural language to SQL, local execution |
| [Music Generator](AI_AGENTS/MusicGenrator_Agent) | AI-powered music generation |
| [Chat with Gmail](AI_AGENTS/chat_with_gmail) | LLM-powered Gmail integration and chat |
| [AI Meme Generator](AI_AGENTS/AI_Meme_Generator) | Generate memes with AI |
| [Health & Fitness Agent](AI_AGENTS/Health_Fitness_Agent) | Health and fitness guidance agent |
| [Simple Scraping Agent](AI_AGENTS/Simple_ScrapingAgent) | Intelligent web scraping |
| [Journalist Agent](AI_AGENTS/Journalist_Agent) | Automated journalism and article drafting |
| [Chat with Tarots](AI_AGENTS/chat-with-tarots) | Tarot reading and chat |
| [Home Renovation Agent](AI_AGENTS/Home_Renovation_agent) | Home renovation planning with AI |
| [Startup Insight Agent](AI_AGENTS/Startup_Insight_Agent) | Startup and business insights |
| [Movie Production Agent](AI_AGENTS/movie_production_agent) | Movie and video production assistance |
| [Chat YouTube](AI_AGENTS/chat_youtube) | Chat over YouTube content |
| [Local Llama Agent](AI_AGENTS/LocalLama_Agent) | Local LLM agent with Llama |
| [LinkedIn Roster](AI_AGENTS/LINKEDIN_ROSTER) | LinkedIn-related agent |

</details>

<details open>
<summary><b>RAG Applications</b> - Click to expand</summary>

| Project | Description |
|---------|-------------|
| [Self-Healing RAG](RAG_systems/SelfHealing_RAG) | Detects poor retrieval, retries search, then generates answer; Azure OpenAI + Pinecone; Streamlit |
| [Adaptive RAG](RAG_systems/Adaptive_RAG) | Query-classified retrieval: vector (FAISS), graph (Neo4j), or hybrid; Azure OpenAI; Streamlit + CLI |
| [Multi-Agent RAG](RAG_systems/MultiAgent_RAG) | Retriever → Reasoning → Verification agents; SentenceTransformers + FAISS; Streamlit UI |
| [Context Compression RAG](RAG_systems/Context_Compression_RAG) | Top-20 retrieval → LLM compression → top-5 → answer; Azure OpenAI, FAISS, Streamlit |
| [Hybrid Graph RAG](RAG_systems/HybridGraph_RAG) | Vector (Pinecone) + graph (Neo4j) retrieval; Streamlit |
| [Page-Indexed RAG](RAGs/PageIndexed_RAG) | PDF RAG with page-level retrieval and source attribution |
| [PDF RAG](RAGs/PDF_RAG) | Chat and query over PDF documents |
| [GraphRAG Papers](RAGs/GraphRAG_Papers) | Graph-based RAG over academic papers (Neo4j + Next.js) |

Advanced retrieval-augmented generation: document indexing, semantic search, context-aware QA, multi-source integration. Many RAG systems use the repo root `.env` for Azure/Neo4j credentials.

</details>

<details open>
<summary><b>Machine Learning Algorithms</b> - Click to expand</summary>

| Project | Description |
|---------|-------------|
| [Supervised Learning](ALL_MachineLearning_Algos/Supervised_Learning) | Regression, classification, XGBoost, AdaBoost, Gradient Boosting |
| [Unsupervised Learning](ALL_MachineLearning_Algos/Unsupervised_Learning) | PCA, clustering, dimensionality reduction |

From-scratch implementations: Linear/Logistic Regression, Decision Trees, SVM, K-Means, PCA, optimization (Gradient Descent, Adam, RMSprop).

</details>

<details>
<summary><b>Fine-Tuning Projects</b> - Click to expand</summary>

| Project | Description |
|---------|-------------|
| [Local Trained Code Model](FineTunning_Projects/LocalTrained_CodeModel) | LoRA fine-tuning for code generation |
| [Legal Docs Summarization](FineTunning_Projects/LegalDocs_Summarization) | Domain fine-tuning for legal document summarization |

LoRA, parameter-efficient fine-tuning, training pipelines and evaluation.

</details>

<details>
<summary><b>Voice Agents</b> - Click to expand</summary>

| Project | Description |
|---------|-------------|
| [Voice Tutor Agent](VOICE_AGENTS/VoiceTutor_Agent) | Voice-powered tutoring |
| [Voice ML Interview Coach](VOICE_AGENTS/VoiceMLInterview_Coach) | Interview practice with voice AI |
| [Role-Based Voice Agent](VOICE_AGENTS/RoleBased_VoiceAgent) | Role-based voice assistants |
| [VOICE RAG](VOICE_AGENTS/VOICE_RAG) | Voice interface to RAG systems |
| [Tour Agent](VOICE_AGENTS/Tour_Agent) | Voice-guided tour agent |
| [Websupport Voice Agent](VOICE_AGENTS/Websupport_voiceAgent) | Customer support voice agent |

</details>

<details>
<summary><b>MCP Agents</b> - Click to expand</summary>

| Project | Description |
|---------|-------------|
| [GitHub MCP Agent](MCP_AGENTS/github_mcp_agent) | GitHub integration via MCP |
| [Browser MCP Agent](MCP_AGENTS/Browser_mcp_agent) | Browser automation via MCP |
| [Git Q&A MCP Agent](MCP_AGENTS/Git_Q&A_MCPagent) | Git repository Q&A with MCP |

</details>

<details>
<summary><b>N8N Automation</b> - Click to expand</summary>

| Project | Description |
|---------|-------------|
| [AI ChatBot Searching Web](N8N_Automation_WorkFlows/AIChatBot_searchingWeb) | Conversational agent with tool use (n8n + LangChain) |
| [Agent Current Weather Wikipedia](N8N_Automation_WorkFlows/Agent_CurrentWeather_Wikipedia) | Weather and Wikipedia agent |
| [Chat Research Agent](N8N_Automation_WorkFlows/ChatResearch_agent) | Research workflow agent |
| [Deepseek AI Researcher](N8N_Automation_WorkFlows/DeepseekAI_Researcher) | AI researcher workflow |
| [OpenAI Scrapper](N8N_Automation_WorkFlows/OPENAI_Scrapper) | Web scraping with OpenAI |
| [Easy Image Captioning](N8N_Automation_WorkFlows/EasyImageCaptioning_openai) | Image captioning workflow |
| [Spot Workplace Discrimination](N8N_Automation_WorkFlows/SpotWorkplace_Diiscrimination) | Workplace pattern detection |
| [URL/HTML to Markdown](N8N_Automation_WorkFlows/URLHTML2Markdown) | Convert URL/HTML to Markdown |
| [Video Narrating](N8N_Automation_WorkFlows/VideoNarrating) | Video narration with AI |

</details>

<details>
<summary><b>Data Science Projects</b> - Click to expand</summary>

| Project | Description |
|---------|-------------|
| [ChurnSense](DS_PROJECTS/ChurnSense) | Churn analysis and user segmentation |
| [Synthetic Data Factory](DS_PROJECTS/Synthetic_Data_Factory) | Synthetic data generation pipeline |
| [MemoryBase Chatbot](DS_PROJECTS/MemoryBase_Chatbot) | Chatbot with persistent memory (Pinecone, Redis) |
| [Multi-Agent Orchestrator (A2A)](DS_PROJECTS/Multi-agent-orchestrator-system-with-standardized-A2A-Protocol) | Multi-agent orchestration with A2A protocol |
| [Daily News Incremental Model](DS_PROJECTS/DailyNews_IncrementalModel) | Incremental learning for news |
| [AI Expense Management](DS_PROJECTS/AIExpense-ManagementSystem) | Expense management system with AI |

</details>

---

## Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python3 --version

# pip package manager
pip3 --version
```

### Installation Steps

<table>
<tr>
<td>

**Step 1: Clone the Repository**
```bash
git clone https://github.com/Anujpatel04/Anuj-AI-ML-Lab.git
cd Anuj-AI-ML-Lab
```

</td>
</tr>
<tr>
<td>

**Step 2: Set up environment**
```bash
# Copy .env.example to .env and add your API keys
cp .env.example .env
# Edit .env with your keys (e.g. OPENAI_API_KEY)
```

</td>
</tr>
<tr>
<td>

**Step 3: Navigate to Project**
```bash
cd <project_folder>
```

</td>
</tr>
<tr>
<td>

**Step 4: Install Dependencies**
```bash
python3 -m pip install -r requirements.txt
```

</td>
</tr>
<tr>
<td>

**Step 5: Follow Project README**
```bash
# Each project has specific setup instructions
cat README.md
```

</td>
</tr>
</table>

---
## Contributing

<div align="center">

We welcome contributions from the community!

[![Contributors](https://img.shields.io/badge/Contribute-Welcome-brightgreen?style=for-the-badge)](CONTRIBUTING.md)

</div>

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Code of Conduct

This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org). See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

<div align="center">

### Show Your Support

If you find this project useful, please consider giving it a star!

---

**Made with care by Anuj**
</div>
