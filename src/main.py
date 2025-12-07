from smolagents import CodeAgent
from agents import inference_model, web_agent, retriever_agent, image_agent
from retriever import prepare_source_docs, create_vector_db



def main():
    model = inference_model()
    source_docs = prepare_source_docs()
    vector_db = create_vector_db(source_docs)

    managed_web_agent = web_agent(model)
    managed_retriever_agent = retriever_agent(model, vector_db)
    managed_image_generation_agent = image_agent(model)

    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[managed_web_agent, managed_retriever_agent, managed_image_generation_agent],
        additional_authorized_imports=["time", "datetime", "PIL"],
    )

    manager_agent.run("How many years ago was Stripe founded?")


if __name__ == "__main__":
    main()
