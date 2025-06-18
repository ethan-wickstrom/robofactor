# 🤖 Resting Agent

> Autonomous REST API Generator for Laravel Applications

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Resting Agent is an AI-powered tool that automatically generates complete RESTful APIs for Laravel applications from natural language descriptions. Simply describe what you want, and Resting Agent will create models, migrations, controllers, routes, validation, and tests - all following Laravel best practices.

## ✨ Features

- **Natural Language to Code**: Describe your API in plain English
- **Complete API Generation**: Creates all necessary Laravel components
- **Production-Ready Code**: Follows Laravel 10+ conventions and PSR-12 standards
- **Smart Planning**: Understands dependencies and generates code in the correct order
- **Test Coverage**: Automatically generates feature tests for your API
- **Multiple LLM Support**: Works with OpenAI, Anthropic, and local models

## 📋 Prerequisites

- Python 3.12 or higher
- Laravel project (8.x or higher)
- API key for your chosen LLM (OpenAI, Anthropic, etc.)

## 🚀 Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/resting-agent.git
cd resting-agent

# Install with UV
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/resting-agent.git
cd resting-agent

# Install in development mode
pip install -e .
```

## 🔧 Configuration

Set your API keys as environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# For Anthropic
export ANTHROPIC_API_KEY="your-api-key-here"
```

## 📖 Usage

### Basic Usage

Generate a simple blog API:

```bash
resting-agent "Create a blog API with posts and comments" --project /path/to/laravel
```

### Advanced Options

```bash
# Use a specific model
resting-agent "E-commerce API with products and orders" \
  --project /path/to/laravel \
  --model claude-3-opus \
  --temperature 0.7

# Dry run to preview actions
resting-agent "User authentication API" \
  --project /path/to/laravel \
  --dry-run

# Verbose output for debugging
resting-agent "Task management API" \
  --project /path/to/laravel \
  --verbose
```

### Available Commands

```bash
# Main generation command
resting-agent [INTENT] [OPTIONS]

# Show version
resting-agent version

# List supported models
resting-agent models

# Get help
resting-agent --help
```

### Command Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--project` | `-p` | Path to Laravel project | Current directory |
| `--model` | `-m` | LLM model to use | `openai/gpt-4o-mini` |
| `--temperature` | `-t` | Generation temperature (0.0-2.0) | `0.7` |
| `--max-tokens` | | Maximum tokens for generation | `4096` |
| `--cache` | | Enable response caching | `True` |
| `--no-cache` | | Disable response caching | |
| `--retries` | | Number of API retries | `3` |
| `--verbose` | `-v` | Enable verbose output | `False` |
| `--dry-run` | | Preview without executing | `False` |

## 🎯 Examples

### Blog Platform API

```bash
resting-agent "Create a blog platform API with:
- Posts with title, content, slug, and published status
- Categories with hierarchical structure
- Tags for posts
- Comments with nested replies
- User authentication and authorization"
```

### E-commerce API

```bash
resting-agent "Build an e-commerce API featuring:
- Products with variants (size, color)
- Shopping cart functionality
- Order management with status tracking
- Payment integration hooks
- Customer reviews and ratings"
```

### Project Management API

```bash
resting-agent "Design a project management API with:
- Projects with team members
- Tasks with due dates and priorities
- Time tracking entries
- File attachments
- Activity feed and notifications"
```

## 🏗️ What Gets Generated

For each API, Resting Agent creates:

1. **Database Layer**
   - Eloquent models with relationships
   - Database migrations with indexes
   - Model factories for testing

2. **API Layer**
   - RESTful controllers with all CRUD operations
   - Form requests for validation
   - API resources for response formatting
   - Route definitions with proper middleware

3. **Testing**
   - Feature tests for all endpoints
   - Test data seeders
   - Authentication test scenarios

4. **Documentation**
   - API route list
   - Request/response examples
   - Validation rules documentation

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
make install-dev

# Run tests
make test

# Run linting
make lint

# Format code
make format

# Type checking
make type-check

# Run all checks
make check
```

### Project Structure

```
resting-agent/
├── src/resting_agent/
│   ├── core/           # Core functionality
│   │   ├── args.py     # CLI argument processing
│   │   ├── cli.py      # CLI interface
│   │   ├── config.py   # Configuration models
│   │   └── types.py    # Type definitions
│   ├── services/       # Service layer
│   ├── agent.py        # Main agent logic
│   ├── executors.py    # Action executors
│   └── signatures.py   # DSPy signatures
├── tests/              # Test suite
├── Makefile           # Development tasks
└── pyproject.toml     # Project configuration
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [DSPy](https://github.com/stanfordnlp/dspy) for structured AI interactions
- CLI powered by [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/)
- Inspired by the Laravel community's commitment to developer experience

## 🐛 Troubleshooting

### Common Issues

**Model API Key Missing**
```bash
# Set your API key
export OPENAI_API_KEY="sk-..."
# Or for Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Laravel Project Not Found**
```bash
# Ensure you're pointing to a valid Laravel project
resting-agent "..." --project /absolute/path/to/laravel
```

**Permission Errors**
```bash
# Ensure the Laravel project is writable
chmod -R 775 /path/to/laravel/app
chmod -R 775 /path/to/laravel/database
```

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our server](https://discord.gg/example)
- 🐞 Issues: [GitHub Issues](https://github.com/yourusername/resting-agent/issues)