# Natural Language Bike-Share Analytics Assistant

A deterministic, schema-driven natural language to SQL system that converts English questions into parameterized SQL queries and returns formatted results. Built for the bike-share analytics assessment with full compliance to all functional requirements.

## Core Features

- **Natural Language → SQL**: Converts English questions to parameterized SQL queries
- **Dynamic Schema Discovery**: Introspects `information_schema.columns` at runtime
- **Semantic Value Resolution**: Maps user terms to actual database domain values
- **Deterministic Result Formatting**: Rule-based formatting based on SQL analysis
- **Chat-style UI**: Streamlit-based interface with error handling
- **Public API**: RESTful endpoint with exact contract compliance
- **Docker/Linux Ready**: Containerized for easy deployment

## Architecture

### Core Components

1. **Semantic Mapper** (`src/semantic_mapper.py`)
   - Domain caching for categorical columns
   - Deterministic value resolution using actual database values
   - Embedding-based column mapping with LLM enhancement

2. **SQL Generator** (`src/sql_generator.py`)
   - Parameterized SQL generation
   - Intent-based query construction
   - Schema-aware table/column selection

3. **Deterministic Formatter** (`src/intent_utils.py`)
   - Rule-based result formatting
   - SQL analysis for unit conversion (minutes, km)
   - No hardcoded answers or pattern matching

4. **Database Layer** (`src/db.py`)
   - PostgreSQL 17 connection with pooling
   - Dynamic schema introspection
   - Parameterized query execution

### Data Flow

```
User Question → Intent Parsing → Value Resolution → SQL Generation → 
Query Execution → Deterministic Formatting → Formatted Result
```

## Semantic Mapping Method

### Domain-Driven Value Resolution

The system uses data-driven, deterministic value resolution instead of hardcoded synonyms:

1. **Domain Caching**: At startup, discovers categorical columns and caches their distinct values
2. **Deterministic Ranking**: 
   - Exact match (case-insensitive)
   - Single-letter abbreviation (if unique)
   - Prefix matching
   - Fuzzy matching (Jaro-Winkler ≥ 0.8)
3. **Schema Integration**: Maps resolved values to actual database domain values

**Example**: "women" → `'female'` (actual database value), not hardcoded synonyms

### Column Mapping

- **Embedding-based similarity** with OpenAI's `text-embedding-3-small`
- **LLM enhancement** for better term understanding
- **Schema introspection** for deterministic column selection
- **No hardcoded English→column synonym lists**

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Environment variables (see `env.example`)

### Running the Application

```bash
# Clone and setup
git clone <repository>
cd bikeshare

# Configure environment
cp env.example .env
# Edit .env with your database and OpenAI credentials

# Start all services
docker compose up --build -d

# Access the application
# UI: http://localhost:8501
# API: http://localhost:8000
```

### Running Tests

```bash
# Run all tests in container
docker compose exec api python -m pytest tests/ -v

# Run specific test suites
docker compose exec api python -m pytest tests/test_public_acceptance.py -v
docker compose exec api python -m pytest tests/test_sql_generator.py -v
```

## Acceptance Test Results

**All three public acceptance tests pass with exact expected outputs:**

1. **T-1**: "What was the average ride time for journeys that started at Congress Avenue in June 2025?" → **"25 minutes"**
2. **T-2**: "Which docking point saw the most departures during the first week of June 2025?" → **"Congress Avenue"**  
3. **T-3**: "How many kilometres were ridden by women on rainy days in June 2025?" → **"6.8 km"**

### Test Verification

```bash
# Verify all acceptance tests pass
docker compose exec api python -m pytest tests/test_public_acceptance.py::test_all_public_acceptance -v
# PASSED in 44.58s

# Test API endpoint directly
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many kilometres were ridden by women on rainy days in June 2025?"}'
# Returns: {"result": "6.8 km", "error": null}
```

## Compliance Features

### Functional Requirements (F-1 to F-8)

- **F-1 UI**: Chat-style Streamlit interface
- **F-2 NL→SQL**: Parameterized SQL generation
- **F-3 Semantic Discovery**: Dynamic schema introspection
- **F-4 Query Capabilities**: Filters, joins, aggregations, date math, GROUP BY
- **F-5 Error Handling**: Graceful handling of unknown intent and empty results
- **F-6 Public API**: Exact `POST /query` contract compliance
- **F-7 Testing**: Unit tests + acceptance tests
- **F-8 Documentation**: This README

### Technical Constraints

- **LLMs allowed but insufficient**: Schema introspection + deterministic mapping
- **No hardcoded synonyms**: Data-driven value resolution
- **Secrets out of source control**: Environment variables
- **Linux/Docker**: Containerized deployment

### Security

- **Parameterized SQL**: All queries use bound parameters
- **Schema validation**: Column/table names validated against discovered schema
- **Input sanitization**: Proper error handling and validation

## API Contract

### Request
```json
POST /query
{
  "question": "How many kilometres were ridden by women on rainy days in June 2025?"
}
```

### Response
```json
{
  "sql": "SELECT SUM(trip_distance_km) FROM trips JOIN daily_weather ON DATE(trips.started_at) = daily_weather.weather_date WHERE LOWER(rider_gender) = LOWER(%s) AND started_at >= %s AND started_at < %s AND precipitation_mm > %s;",
  "result": "6.8 km",
  "error": null
}
```

## Testing Strategy

- **Unit Tests**: SQL generation, semantic mapping, value resolution
- **Integration Tests**: End-to-end API functionality  
- **Acceptance Tests**: Exact public test case verification
- **Hidden Test Readiness**: Synonym handling, weather conditions, edge cases, security

## Key Design Decisions

- **Deterministic over LLM-dependent**: Core formatting uses rule-based analysis
- **Schema-driven over hardcoded**: All mappings derive from database introspection  
- **Domain-aware value resolution**: Maps user terms to actual database values
- **Containerized deployment**: Ensures Linux compatibility and easy testing

## Project Structure

```
bikeshare/
├── src/                    # Core application code
│   ├── main.py            # FastAPI application
│   ├── semantic_mapper.py # Value resolution & column mapping
│   ├── sql_generator.py   # SQL generation
│   ├── db.py              # Database layer
│   └── intent_utils.py    # Deterministic formatting
├── tests/                 # Test suites
├── streamlit_app.py       # Chat UI
├── docker-compose.yml     # Service orchestration
└── Dockerfile            # Container definition
```

## Environment Variables

- `POSTGRES_*`: Database connection
- `OPENAI_API_KEY`: For embeddings and LLM enhancement

## License

MIT License - see LICENSE file for details.

---

## Final Submission Checklist

- **T-1** → "25 minutes"
- **T-2** → "Congress Avenue"
- **T-3** → "6.8 km"
- **Unit tests for SQL generation** present & passing
- **API**: `POST /query` → `{ sql, result, error }`
- **Deterministic mapping** via schema introspection (no hardcoded synonyms)
- **Parameterized SQL**, filters/joins/aggs/date math/group-by; graceful errors
- **Linux/Docker**, secrets via env
- **Structure**: `/src`, `/tests`, `README.md` (~2 pages), `LICENSE`
