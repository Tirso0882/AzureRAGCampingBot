import contextlib
import multiprocessing
import os
from pathlib import Path
from pprint import pprint

import pandas as pd
from azure.ai.evaluation import GroundednessEvaluator, evaluate
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from chat_with_products import chat_with_products
from config import ASSET_PATH
from dotenv import load_dotenv

load_dotenv()


project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

connection = project.connections.get_default(connection_type=ConnectionType.AZURE_OPEN_AI, include_credentials=True)

evaluator_model = {
    "azure_endpoint": connection.endpoint_url,
    "azure_deployment": os.environ["EVALUATION_MODEL"],
    "api_version": "2024-06-01",
    "api_key": connection.key,
}

groundedness = GroundednessEvaluator(evaluator_model)

def evaluate_chat_with_products(query):
    """Evaluate the chat_with_products function with a given query.
    
    Calls the chat_with_products function with a user query and formats 
    the response and context for evaluation by the groundedness evaluator.
    
    Args:
        query: A string containing the user's query about camping products.
    
    Returns:
        A dictionary containing the assistant's response text and the 
        grounding data context used to generate the response.
    """
    response = chat_with_products(messages=[{"role": "user", "content": query}])
    return {"response": response["message"].content, "context": response["context"]["grounding_data"]}


if __name__ == "__main__":
    
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)

    result = evaluate(
        data=Path(ASSET_PATH) / "chat_eval_data.jsonl",
        target=evaluate_chat_with_products,
        evaluation_name="evaluate_chat_with_products",
        evaluators={
            "groundedness": groundedness,
        },
        evaluator_config={
            "default": {
                "query": {"${data.query}"},
                "response": {"${target.response}"},
                "context": {"${target.context}"},
            }
        },
        azure_ai_project=project.scope,
        output_path=Path(ASSET_PATH) / "./myevalresults.json",
    )

    tabular_result = pd.DataFrame(result.get("rows"))

    pprint("-----Summarized Metrics-----")
    pprint(result["metrics"])
    pprint("-----Tabular Result-----")
    pprint(tabular_result)
    pprint(f"View evaluation results in AI Studio: {result['studio_url']}")