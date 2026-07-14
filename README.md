# Clinical Trial Matching Agentic AI 



This is a Flask backend application with an agentic AI system that analyzes patient-trial matching using multiple specialized agents. The system simulates Google ADK (Agent Development Kit) style agent orchestration for clinical trial eligibility analysis.



## Features



- **Agentic AI Architecture**: Multi-agent system with 7 specialized agents (including reporting)

- **Clinical Trial Matching**: Automated patient-trial eligibility analysis

- **RESTful API**: Clean endpoints for frontend integration

- **Real-time Processing**: Fast analysis with processing time tracking



## Agent Architecture



The system uses 5 specialized agents that work together:



1. **Data Ingestion Agent** - Validates and ingests patient/trial data

2. **Medical Analysis Agent** - Analyzes medical conditions and lab values

3. **Trial Matching Agent** - Matches patients to eligible clinical trials

4. **Risk Assessment Agent** - Identifies potential risks and contraindications

5. **Recommendation Agent** - Generates clinical recommendations and next steps

6. **Report Agent** - Assembles final report JSON combining patient profile, matches, and explanations



## Setup Instructions



```bash

python -m venv venv

venv\Scripts\activate  # On Windows

# or

source venv/bin/activate  # On macOS/Linux

```



### 2. Install Dependencies



```bash

pip install -r requirements.txt

```



### 3. Configure Environment Variables



Copy `.env.example` to `.env` and update the values:



```bash

cp .env.example .env

```



Edit `.env` and add your OpenAI API key:

```

OPENAI_API_KEY=your_openai_api_key_here

SECRET_KEY=your_secret_key_here

```



### 4. Run the Application



```bash

python app.py

```



The server will start at `http://localhost:5000`




## Key Features



- ✅ Flask REST API with CORS support

- ✅ LangChain integration for Agentic AI

- ✅ OpenAI GPT-4 support

- ✅ Error handling and validation

- ✅ Multi-environment configuration

- ✅ Extensible tool system for agents




This project is part of CodeCrusaders team.
