# ruff: noqa: ANN201, ANN001

import logging
import os
import pathlib
import sys

from azure.ai.inference.tracing import AIInferenceInstrumentor
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

ASSET_PATH = pathlib.Path(__file__).parent.resolve() / "assets"

logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_logger(module_name):
    """Create a named logger for a specific module.
    
    Args:
        module_name: Name of the module requesting the logger.
        
    Returns:
        A configured Logger instance with the namespace "app.{module_name}".
    """
    return logging.getLogger(f"app.{module_name}")


def enable_telemetry(log_to_project: bool = False):
    """Configure OpenTelemetry instrumentation for the application.
    
    Sets up instrumentation for Azure AI services and optionally
    configures logging to Azure Application Insights within the
    AI project.
    
    Args:
        log_to_project: If True, telemetry will be sent to the 
                        Application Insights instance connected
                        to the Azure AI project. Default is False.
    """
    AIInferenceInstrumentor().instrument()

    # enable logging message contents
    os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = "true"

    if log_to_project:
        from azure.monitor.opentelemetry import configure_azure_monitor

        project = AIProjectClient.from_connection_string(
            conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
        )
        tracing_link = f"https://ai.azure.com/tracing?wsid=/subscriptions/{project.scope['subscription_id']}/resourceGroups/{project.scope['resource_group_name']}/providers/Microsoft.MachineLearningServices/workspaces/{project.scope['project_name']}"
        application_insights_connection_string = project.telemetry.get_connection_string()
        if not application_insights_connection_string:
            logger.warning(
                "No application insights configured, telemetry will not be logged to project. Add application insights at:"
            )
            logger.warning(tracing_link)

            return

        configure_azure_monitor(connection_string=application_insights_connection_string)
        logger.info("Enabled telemetry logging to project, view traces at:")
        logger.info(tracing_link)