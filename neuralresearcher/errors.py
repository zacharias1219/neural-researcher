class NeuralResearcherError(Exception):
    """Base exception for all neuralresearcher errors."""
    pass

class ConfigError(NeuralResearcherError):
    """Configuration related errors."""
    pass

class LLMError(NeuralResearcherError):
    """Errors arising from LLM interactions."""
    pass

class ToolError(NeuralResearcherError):
    """Errors arising during tool execution."""
    pass

class SchemaError(NeuralResearcherError):
    """Errors arising from unexpected structured output schema."""
    pass

class WorkflowError(NeuralResearcherError):
    """Errors arising during orchestrator transitions or state logic."""
    pass
