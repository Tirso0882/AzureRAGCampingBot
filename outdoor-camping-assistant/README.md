# Outdoor Camping Assistant

A RAG (Retrieval Augmented Generation) based AI assistant that helps users find and learn about outdoor camping products. The assistant leverages Azure AI Search with vector search capabilities to provide grounded, accurate product recommendations.

## Overview

This project implements a conversational AI assistant specialized in outdoor camping equipment. It uses:

- Azure AI Search with vector embeddings for semantic search
- Azure OpenAI Service for chat completions and embeddings
- Grounded responses based on the product catalog
- Evaluation capabilities to assess response quality

## Architecture

The system follows this workflow:

1. User queries are processed to extract search intent
2. Vector embeddings are created for the query
3. Azure AI Search finds relevant product documents using hybrid search (vector + keyword)
4. Retrieved product information is used to ground the AI response
5. The assistant provides contextually relevant product recommendations

## Setup

### Prerequisites

- Python 3.11+
- Azure subscription with access to:
  - Azure AI Studio project
  - Azure OpenAI Service
  - Azure AI Search

### Environment Variables

Create a `.env` file with the following variables:

```
AIPROJECT_CONNECTION_STRING=<ai_project_connection_string>
CHAT_MODEL=<chat_model_deployment_name>
INTENT_MAPPING_MODEL=<intent_mapping_model_deployment_name>
EMBEDDINGS_MODEL=<embeddings_model_deployment_name>
EVALUATION_MODEL=<evaluation_model_deployment_name>
AISEARCH_INDEX_NAME=<search_index_name>
```

### Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

### Creating the Search Index

First, create and populate the search index with product data:

```bash
python create_search_index.py --csv-file assets/products.csv
```

### Running the Assistant

To chat with the assistant:

```bash
python chat_with_products.py --query "I need a tent for 4 people, what would you recommend?"
```

### Evaluation

The evaluation framework assesses the quality of the assistant's responses using the Azure AI Evaluation framework. It focuses on **groundedness** - ensuring responses are factually accurate and derived from the provided product information rather than hallucinated.

<table>
  <tr>
    <td width="200">
      <img src="../screenshots/evaluation_groundedness.png" alt="EVALUATION tool" width="400"/>
    </td>
    <td>
      This average score indicates better groundedness. Since the scale ranges from 1 to 5, a score of 4.15 suggests that the content is generally well-grounded but has room for improvement.
      <h4>Evaluation Analysis - Distribution of Scores</h4>
      <ul>
        <li><strong>Score 1:</strong> Very low groundedness. Only a small number of evaluations received this score.</li>
        <li><strong>Score 2:</strong> Low groundedness. No evaluations received this score.</li>
        <li><strong>Score 3:</strong> Moderate groundedness. A moderate number of evaluations received this score.</li>
        <li><strong>Score 4:</strong> High groundedness. A significant number of evaluations received this score.</li>
        <li><strong>Score 5:</strong> Very high groundedness. The largest number of evaluations received this score.</li>
      </ul>
    </td>
  </tr>
</table>

To evaluate the assistant's performance:

```bash
python evaluate.py
```

This runs a set of test queries from `assets/chat_eval_data.jsonl` and evaluates the groundedness of responses.

## Core Components

### `chat_with_products.py`

The main entry point for the chatbot. Processes user messages, retrieves relevant product documents, and generates grounded responses.

### `get_product_documents.py`

Handles intent mapping from user queries to search queries, generates embeddings, and retrieves relevant product documents from Azure AI Search.

### `create_search_index.py`

Creates and manages the Azure AI Search index with vector search capabilities. Processes product data and generates embeddings.

### `evaluate.py`

Evaluates the quality of assistant responses using the Azure AI Evaluation framework, focusing on groundedness.

### `config.py`

Contains configuration utilities, logging setup, and telemetry configuration.

## Customization

To add new products to the assistant:
1. Update the `assets/products.csv` file with new product information
2. Run `create_search_index.py` to rebuild the search index

## Telemetry Logging

<table>
  <tr>
    <td width="400">
      <img src="../screenshots/trace_chat_with_products.png" alt="TRACING tool" width="400"/>
    </td>
    <td>
      <strong>Tracing</strong> helps understanding the execution flow, debugging issues, and optimising performance, especially in complex AI workflows. It provides a detailed view of how the application behaves, making it easier to identify bottlenecks, pinpoint errors, and ensure the application is working as intended.
    </td>
  </tr>
</table>

To enable telemetry during development:

```bash
python chat_with_products.py --query "query goes here" --enable-telemetry
```