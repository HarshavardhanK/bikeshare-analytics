# Bikeshare Analytics Assistant

A natural-language analytics assistant for bike-share PostgreSQL database that converts natural language queries to SQL and provides insights.

## Features

- Natural language to SQL conversion using OpenAI
- Semantic mapping for user-friendly column names
- Real-time database querying
- Structured logging with request tracing
- Production-ready Docker deployment
- Health checks and monitoring endpoints

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- PostgreSQL database (Azure PostgreSQL recommended)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bikeshare
   ```

2. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/ping
   - Readiness Check: http://localhost:8000/ready

## Production Deployment

### Pre-flight Checklist

Before deploying to production, ensure:

- [ ] All environment variables are configured
- [ ] Database is accessible with SSL
- [ ] OpenAI API key is valid
- [ ] Docker image builds successfully
- [ ] Health checks pass

### Automated Deployment

Use the provided deployment script:

```bash
./deploy.sh
```

This script will:
- Validate environment variables
- Build the Docker image
- Run security scans (if Trivy is available)
- Start services with health checks
- Verify deployment success

### Manual Deployment

1. **Build the image**
   ```bash
   docker build -t bikeshare-analytics:latest .
   ```

2. **Start services**
   ```bash
   docker-compose up -d
   ```

3. **Verify deployment**
   ```bash
   curl http://localhost:8000/ping
   curl http://localhost:8000/ready
   ```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POSTGRES_DB` | Database name | - | Yes |
| `POSTGRES_USER` | Database user | - | Yes |
| `POSTGRES_PASSWORD` | Database password | - | Yes |
| `POSTGRES_HOST` | Database host | - | Yes |
| `POSTGRES_PORT` | Database port | 5432 | No |
| `PGSSLMODE` | SSL mode for database | require | No |
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `GRADER_MODE` | Enable grader mode | 0 | No |
| `LOG_LEVEL` | Logging level | INFO | No |

### Database Configuration

For Azure PostgreSQL:
- Set `PGSSLMODE=require`
- Ensure firewall rules allow your deployment IP
- Use connection pooling (configured automatically)

### SSL Configuration

The application automatically configures SSL for database connections when `PGSSLMODE` is set to `require`.

## API Endpoints

### Health Checks

- `GET /ping` - Basic health check
- `GET /ready` - Readiness check (includes database connectivity)

### Main Endpoint

- `POST /query` - Process natural language queries

Example request:
```json
{
  "question": "How many female riders are there?"
}
```

Example response:
```json
{
  "sql": "SELECT COUNT(*) FROM trips WHERE rider_gender ILIKE 'female'",
  "result": [[1500]],
  "error": null
}
```

## Monitoring and Logging

### Structured Logging

The application uses structured JSON logging with:
- Request IDs for tracing
- Timestamp in ISO format
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Contextual information

### Health Monitoring

Monitor the application using:
- `/ping` endpoint for basic health
- `/ready` endpoint for full readiness
- Docker health checks
- Application logs

### Performance Metrics

Key metrics to monitor:
- Query response times
- Database connection pool usage
- OpenAI API response times
- Error rates

## Security

### Production Security Features

- Non-root user execution
- SSL/TLS encryption for database
- Environment variable secrets management
- Request ID tracing
- Structured error handling

### Security Best Practices

1. **Never commit secrets to version control**
2. **Use environment variables for all sensitive data**
3. **Enable SSL for database connections**
4. **Regular security scans with Trivy**
5. **Keep base images updated**

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check `POSTGRES_HOST` and credentials
   - Verify SSL configuration
   - Check firewall rules

2. **OpenAI API Errors**
   - Validate API key
   - Check API quota
   - Verify network connectivity

3. **Container Won't Start**
   - Check environment variables
   - Verify Docker image builds
   - Check resource limits

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
docker-compose restart api
```

### Logs

View application logs:
```bash
docker-compose logs -f api
```

View database logs:
```bash
docker-compose logs -f db
```

## Development

### Running Tests

```bash
docker-compose exec api pytest
```

### Code Quality

The project follows these practices:
- Type hints for all functions
- Comprehensive error handling
- Structured logging
- Docker-compatible code

## License

[License information]

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Verify configuration
4. Open an issue with detailed information
