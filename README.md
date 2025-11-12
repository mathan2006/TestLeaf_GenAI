# DeepEval Demo - Multi-Provider LLM Evaluation Service

Enterprise-grade FastAPI service for evaluating LLM outputs using DeepEval metrics with support for **Groq** and **OpenAI** providers.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Node.js dependencies
npm install
```

### 2. Configure Provider

**Choose your provider** by editing `.env`:

#### Option A: Use Groq (Default - Fast & Cost-Effective)
```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
EVAL_MODEL=llama-3.3-70b-versatile
```

#### Option B: Use OpenAI (Stricter Evaluation)
```bash
OPENAI_API_KEY=sk-your_openai_api_key_here
EVAL_MODEL=gpt-4o-mini
```

**See [PROVIDER_SETUP.md](./PROVIDER_SETUP.md) for detailed provider comparison and configuration.**

### 3. Start Services

```bash
# Terminal 1: Start Python evaluation server
python deepeval_server.py

# Terminal 2: Start Node.js API server
npm run dev
```

### 4. Test the Setup

```bash
# Health check
curl http://localhost:8000/health

# Smoke test (runs a fixed example)
curl http://localhost:8000/example

# Get metrics info
curl http://localhost:8000/metrics-info
```

---

## 📊 Supported Metrics

All metrics run with **strict_mode=True** for rigorous evaluation:

| Metric | Description | Required Fields | Use Case |
|--------|-------------|-----------------|----------|
| **faithfulness** | Measures if output is grounded in context | `output`, `context` | Detect hallucinations |
| **answer_relevancy** | Measures if answer addresses the query | `query`, `output` | Check relevance |
| **contextual_precision** | Measures retrieval precision | `context`, `output`, `expected_output` | RAG quality |
| **contextual_recall** | Measures retrieval coverage | `context`, `output`, `expected_output` | RAG completeness |

---

## 🔧 API Usage

### Evaluate with Faithfulness

```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Selenium?",
    "context": ["Selenium is a web automation framework for testing web applications."],
    "output": "Selenium is a web automation framework.",
    "metric": "faithfulness"
  }'
```

### Evaluate with Answer Relevancy

```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I write Selenium code?",
    "output": "Here is a basic example: driver.get(\"https://example.com\")",
    "metric": "answer_relevancy"
  }'
```

### Response Format

```json
{
  "metric_name": "faithfulness",
  "score": 0.95,
  "explanation": "Faithfulness (strict mode) measures how well the output aligns with the provided context. Score: 0.95/1.0. Strict mode penalizes any claims not directly supported by context."
}
```

---

## 🎯 Provider Comparison

| Feature | Groq | OpenAI |
|---------|------|--------|
| **Speed** | ⚡ Very Fast | 🐢 Moderate |
| **Cost** | 💰 Low | 💰💰 Higher |
| **Strictness** | 🔍 Good | 🔍🔍 Excellent |
| **Setup** | `EVAL_MODEL=llama-3.3-70b-versatile` | `EVAL_MODEL=gpt-4o-mini` |
| **Best For** | Development, Testing, High Volume | Production, QA, Critical Evaluations |

**Switch providers anytime with zero code changes!** Just update `.env` and restart.

See [PROVIDER_SETUP.md](./PROVIDER_SETUP.md) for complete details.

---

## 📚 Project Structure

```
deepeval-demo/
├── deepeval_server.py          # FastAPI evaluation service (Python)
├── src/
│   ├── index.ts                # Express.js API server (Node.js)
│   ├── routes/
│   │   └── evalRoutes.ts       # Evaluation endpoints
│   ├── services/
│   │   ├── evalClient.ts       # Python service client
│   │   ├── llmClient.ts        # LLM provider client
│   │   └── ragService.ts       # RAG pipeline
│   └── config/
│       └── env.ts              # Environment config
├── requirements.txt            # Python dependencies
├── package.json                # Node.js dependencies
├── .env                        # Environment variables
├── PROVIDER_SETUP.md           # Provider configuration guide
└── postman-collection.json     # API testing collection
```

---

## 🛠️ Architecture

```
┌─────────────────┐
│   Node.js API   │  (Port 3000)
│   Express.js    │  - REST endpoints
└────────┬────────┘  - Business logic
         │
         │ HTTP calls
         ▼
┌─────────────────┐
│  Python Server  │  (Port 8000)
│   FastAPI       │  - DeepEval metrics
│   DeepEval      │  - LLM evaluation
└────────┬────────┘  - Groq/OpenAI support
         │
         │ API calls
         ▼
┌─────────────────┐
│  Groq/OpenAI    │
│      API        │
└─────────────────┘
```

**Why This Architecture?**
- **Separation of Concerns:** Node.js handles business logic, Python handles ML evaluation
- **Language Strengths:** Python for ML/AI libraries, Node.js for web APIs
- **Flexibility:** Easy to scale or replace components independently

---

## 🔑 Environment Variables

### Required (Choose One Provider)

```bash
# For Groq (default)
GROQ_API_KEY=gsk_your_groq_api_key_here
EVAL_MODEL=llama-3.3-70b-versatile

# For OpenAI (alternative)
OPENAI_API_KEY=sk_your_openai_api_key_here
EVAL_MODEL=gpt-4o-mini
```

### Optional

```bash
PORT=3000                              # Node.js server port
DEEPEVAL_URL=http://localhost:8000/eval # Python service URL
```

---

## 📖 Advanced Usage

### Using Both Providers

You can have both API keys configured:

```bash
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk_...
EVAL_MODEL=llama-3.3-70b-versatile  # Active provider
```

Switch providers by changing only `EVAL_MODEL` and restarting.

### Available Models

**Groq:**
- `llama-3.3-70b-versatile` (recommended)
- `llama-3.1-70b-versatile`
- `llama-3.1-8b-instant` (fastest)
- `mixtral-8x7b-32768`
- `gemma2-9b-it`

**OpenAI:**
- `gpt-4o-mini` (recommended)
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Strict Mode

All metrics use `strict_mode=True` by default:
- **Faithfulness:** Penalizes any claims not in context
- **Answer Relevancy:** Penalizes off-topic responses
- **Contextual Precision:** Requires highly relevant context
- **Contextual Recall:** Requires comprehensive coverage

---

## 🧪 Testing

### Run Smoke Test

```bash
curl http://localhost:8000/example
```

This runs a fixed faithfulness evaluation to verify the setup.

### Get Metrics Information

```bash
curl http://localhost:8000/metrics-info
```

Returns detailed info on all available metrics with example requests.

### Import Postman Collection

Import `postman-collection.json` to get pre-configured API requests for all metrics.

---

## 🐛 Troubleshooting

### "GROQ_API_KEY environment variable is required"
- **Fix:** Add `GROQ_API_KEY=gsk_...` to `.env` if using Groq models

### "OPENAI_API_KEY environment variable is required"
- **Fix:** Add `OPENAI_API_KEY=sk_...` to `.env` if using OpenAI models

### Scores Too Lenient
- **Fix:** Switch to OpenAI: `EVAL_MODEL=gpt-4o-mini` for stricter evaluation

### Connection Refused (Port 8000)
- **Fix:** Ensure Python server is running: `python deepeval_server.py`

### Import Errors
- **Fix:** Reinstall dependencies: `pip install -r requirements.txt`

---

## 📦 Dependencies

### Python (requirements.txt)
```
fastapi>=0.104.0
uvicorn>=0.24.0
deepeval>=0.21.0
python-dotenv>=1.0.0
openai>=1.0.0
```

### Node.js (package.json)
```json
{
  "dependencies": {
    "express": "^4.18.2",
    "dotenv": "^16.3.1",
    "axios": "^1.6.0"
  },
  "devDependencies": {
    "typescript": "^5.3.3",
    "ts-node": "^10.9.2",
    "@types/express": "^4.17.21"
  }
}
```

---

## 🚀 Production Deployment

### Recommended Setup

1. **Development/Testing:** Use Groq for fast iteration
2. **Production:** Use OpenAI for maximum accuracy
3. **CI/CD:** Use Groq for automated tests (cost-effective)
4. **Critical QA:** Use OpenAI for final validation

### Environment-Based Configuration

```bash
# .env.development
EVAL_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...

# .env.production
EVAL_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk_...
```

### Docker Deployment

```dockerfile
# Dockerfile.python
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY deepeval_server.py .
CMD ["python", "deepeval_server.py"]
```

```dockerfile
# Dockerfile.node
FROM node:20-slim
WORKDIR /app
COPY package*.json .
RUN npm ci --production
COPY . .
RUN npm run build
CMD ["node", "dist/index.js"]
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both Groq and OpenAI providers
5. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🔗 Resources

- **DeepEval Docs:** https://docs.confident-ai.com/
- **Groq API:** https://console.groq.com/
- **OpenAI API:** https://platform.openai.com/
- **Provider Setup Guide:** [PROVIDER_SETUP.md](./PROVIDER_SETUP.md)

---

## ⭐ Key Features

✅ **Dual Provider Support** - Groq and OpenAI with zero code changes  
✅ **Strict Mode** - All metrics use strict evaluation by default  
✅ **4 Core Metrics** - Faithfulness, Relevancy, Precision, Recall  
✅ **FastAPI + Express** - Modern async architecture  
✅ **Production Ready** - Proper error handling, logging, health checks  
✅ **Easy Testing** - Smoke tests, Postman collection, example requests  
✅ **Comprehensive Docs** - Setup guides, API docs, troubleshooting  

---

**Need help?** Check [PROVIDER_SETUP.md](./PROVIDER_SETUP.md) for detailed configuration options!
