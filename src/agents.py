from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, VisitWebpageTool, InferenceClientModel, Tool, load_tool
from retriever import RetrieverTool


def inference_model(model_id: str = "Qwen/Qwen2.5-72B-Instruct") -> InferenceClientModel:
    model = InferenceClientModel(model_id)
    return model


def web_agent(model: InferenceClientModel) -> ToolCallingAgent:
    web_agent = ToolCallingAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model,
        description="Runs web searches for you. Give it your query as an argument.",
        name="search_agent"
    )
    return web_agent


def retriever_agent(model: InferenceClientModel, vector_db) -> ToolCallingAgent:
    huggingface_doc_retriever_tool = RetrieverTool(vector_db)
    retriever_agent = ToolCallingAgent(
        tools=[huggingface_doc_retriever_tool], model=model, max_steps=4,
        name="retriever_agent",
        description="Retrieves documents from the knowledge base for you that are close to the input query. Give it your query as an argument. The knowledge base includes Hugging Face documentation.",
    )
    return retriever_agent


def image_agent(model: InferenceClientModel) -> CodeAgent:
    prompt_generator_tool = Tool.from_space("sergiopaniego/Promptist", name="generator_tool", description="Optimizes user input into model-preferred prompts")
    image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
    image_generation_agent = CodeAgent(
        tools=[prompt_generator_tool, image_generation_tool],
        model=model,
        description="Generates images from text prompts. Give it your prompt as an argument.",
        instructions="\n\nYour final answer MUST BE only the generated image location."
    )
    return image_generation_agent
