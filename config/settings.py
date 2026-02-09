"""
Azure Configuration Settings
============================
Configuration management for Azure Databricks knowledge structuring pipeline.

In Databricks, these values should be stored in:
- Azure Key Vault (secrets)
- Databricks Secrets (linked to Key Vault)
- Unity Catalog (for data governance)
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI Service configuration."""
    endpoint: str
    api_key: str
    api_version: str = "2024-02-01"
    gpt4o_deployment: str = "gpt-4o"
    embedding_deployment: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072


@dataclass
class AzureStorageConfig:
    """Azure Data Lake Storage Gen2 configuration."""
    account_name: str
    container_name: str
    account_key: Optional[str] = None

    @property
    def connection_string(self) -> str:
        return f"abfss://{self.container_name}@{self.account_name}.dfs.core.windows.net"

    @property
    def bronze_path(self) -> str:
        return f"{self.connection_string}/bronze"

    @property
    def silver_path(self) -> str:
        return f"{self.connection_string}/silver"

    @property
    def gold_path(self) -> str:
        return f"{self.connection_string}/gold"


@dataclass
class AzureSearchConfig:
    """Azure AI Search configuration."""
    endpoint: str
    api_key: str
    index_name: str = "knowledge-index"


@dataclass
class CosmosDBConfig:
    """Azure Cosmos DB Gremlin configuration."""
    endpoint: str
    key: str
    database: str = "knowledge-graph"
    graph: str = "entities"


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    openai: AzureOpenAIConfig
    storage: AzureStorageConfig
    search: AzureSearchConfig
    cosmos: CosmosDBConfig

    # Processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 100
    max_retries: int = 3

    # Language settings
    supported_languages: tuple = ("en", "nl")
    default_language: str = "en"


def get_config_from_databricks() -> PipelineConfig:
    """
    Load configuration from Databricks secrets.

    Usage in Databricks notebook:
        config = get_config_from_databricks()
    """
    try:
        # In Databricks, use dbutils.secrets
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        dbutils = spark._jvm.com.databricks.service.DBUtils(spark)

        def get_secret(scope: str, key: str) -> str:
            return dbutils.secrets.get(scope=scope, key=key)

    except Exception:
        # Local development fallback - use environment variables
        def get_secret(scope: str, key: str) -> str:
            env_key = f"{scope.upper()}_{key.upper()}"
            return os.getenv(env_key, "")

    return PipelineConfig(
        openai=AzureOpenAIConfig(
            endpoint=get_secret("azure-openai", "endpoint"),
            api_key=get_secret("azure-openai", "api-key"),
            gpt4o_deployment=get_secret("azure-openai", "gpt4o-deployment") or "gpt-4o",
            embedding_deployment=get_secret("azure-openai", "embedding-deployment") or "text-embedding-3-large",
        ),
        storage=AzureStorageConfig(
            account_name=get_secret("azure-storage", "account-name"),
            container_name=get_secret("azure-storage", "container-name"),
        ),
        search=AzureSearchConfig(
            endpoint=get_secret("azure-search", "endpoint"),
            api_key=get_secret("azure-search", "api-key"),
        ),
        cosmos=CosmosDBConfig(
            endpoint=get_secret("azure-cosmos", "endpoint"),
            key=get_secret("azure-cosmos", "key"),
        ),
    )


def get_config_from_env() -> PipelineConfig:
    """
    Load configuration from environment variables.

    For local development, create a .env file with:
        AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
        AZURE_OPENAI_API_KEY=xxx
        ...
    """
    from dotenv import load_dotenv
    load_dotenv()

    return PipelineConfig(
        openai=AzureOpenAIConfig(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            gpt4o_deployment=os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT", "gpt-4o"),
            embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
        ),
        storage=AzureStorageConfig(
            account_name=os.getenv("AZURE_STORAGE_ACCOUNT_NAME", ""),
            container_name=os.getenv("AZURE_STORAGE_CONTAINER_NAME", "knowledge-lake"),
        ),
        search=AzureSearchConfig(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
            api_key=os.getenv("AZURE_SEARCH_API_KEY", ""),
        ),
        cosmos=CosmosDBConfig(
            endpoint=os.getenv("AZURE_COSMOS_ENDPOINT", ""),
            key=os.getenv("AZURE_COSMOS_KEY", ""),
        ),
    )
