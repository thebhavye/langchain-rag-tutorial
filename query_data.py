import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from providers import chat, embeddings
from langchain.prompts import ChatPromptTemplate
import os
import neatlogs
import medic_runtime as medic

# Traced into neatlogs with the release and environment Medic needs (added by agent-medic init).
if not os.environ.get("MEDIC_RUNNER"):          # Medic's own runs initialize neatlogs themselves
    neatlogs.init(workflow_name="rag-tutorial",
                  tags=["env:" + os.environ.get("MEDIC_ENV", "prod"),
                        "release:" + os.environ.get("GIT_SHA", "dev")[:12]])

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


NO_RESULTS = "Unable to find matching results."
MIN_RELEVANCE = 0.6        # calibrated for Gemini embeddings (OpenAI ada-002 scores run higher: 0.7)


@medic.surface("retrieve")
def retrieve(query_text: str):
    # Prepare the DB.
    embedding_function = embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    return db.similarity_search_with_relevance_scores(query_text, k=3)


@medic.entrypoint
def answer(query_text: str, results=None) -> str:
    results = retrieve(query_text) if results is None else results
    if len(results) == 0 or results[0][1] < medic.value("min_relevance", MIN_RELEVANCE):
        return NO_RESULTS

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(medic.value("prompt_template", PROMPT_TEMPLATE))
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = chat()
    return model.predict(prompt)


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    results = retrieve(query_text)
    response_text = answer(query_text, results)
    if response_text == NO_RESULTS:
        print(NO_RESULTS)
        return

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
