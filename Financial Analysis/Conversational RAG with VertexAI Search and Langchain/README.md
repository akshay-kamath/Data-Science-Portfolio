# Conversational Document Q&A with Vertex AI Search and LangChain

## Overview

This project demonstrates how to build a sophisticated, conversational question-answering (Q&A) system over a private document corpus using Google Cloud's **Vertex AI Search** and the **LangChain** framework.

The goal is to create a Retrieval-Augmented Generation (RAG) pipeline that allows users to ask questions in natural language and receive accurate, context-aware answers sourced directly from their documents. The system is powered by the Gemini family of models and supports conversational memory, enabling follow-up questions and a more natural user experience.

---

## Key Features

-   **Enterprise-Grade Retrieval**: Leverages **Vertex AI Search** for scalable and secure document indexing, embedding, and retrieval, forming the backbone of the RAG system.
-   **Orchestration with LangChain**: Uses the powerful LangChain framework to seamlessly connect the language model (Gemini) with the document retriever.
-   **Conversational Memory**: Implements a `ConversationalRetrievalChain` with `ConversationBufferMemory` to maintain the context of a dialogue, allowing for intuitive follow-up questions.
-   **Source-Cited Answers**: Includes a `RetrievalQAWithSourcesChain` to ensure that answers are not only accurate but also verifiable, with direct references to the source documents.
-   **Customizable Prompts**: Demonstrates how to use LangChain's `PromptTemplate` to precisely control the model's behavior, ensuring it answers only from the provided context and adheres to specific output formats.
-   **Extractive Answers**: Configures the retriever to pull exact snippets from documents, improving the factual grounding of the LLM's final response.

---

## How It Works

The application follows a classic Retrieval-Augmented Generation (RAG) architecture:

1.  **User Query**: The user asks a question (e.g., "What were alphabet revenues in 2022?").
2.  **Document Retrieval**: The `VertexAISearchRetriever` takes the query and searches the pre-indexed document data store in Vertex AI Search to find the most relevant text chunks or documents.
3.  **Context Augmentation**: LangChain takes the retrieved document chunks and combines them with the user's original question into a detailed prompt.
4.  **Answer Generation**: This augmented prompt is sent to the **Gemini model** via the `ChatVertexAI` wrapper. The model generates a natural language answer based *only* on the context provided by the retrieved documents.
5.  **Maintaining Conversation**: For conversational chains, the user's query and the model's answer are stored in memory, so the model can understand the context of subsequent questions (eg "What about costs and expenses?").

---

## Tech Stack

-   **Cloud Platform**: Google Cloud Platform (GCP)
-   **AI Services**: Vertex AI Search, Vertex AI (Gemini)
-   **Orchestration Framework**: LangChain
-   **Core Language**: Python

---

## Setup and Usage

1.  **Prerequisites**:
    -   A Google Cloud Project.
    -   A **Vertex AI Search data store** populated with the documents you want to query. You can create one in the Google Cloud Console and upload PDFs, HTML files, or other formats.

2.  **Installation**:
    -   Install the necessary Python packages:
        ```bash
        pip install google-cloud-aiplatform google-cloud-discoveryengine langchain-google-vertexai langchain-google-community
        ```

3.  **Configuration**:
    -   In the notebook, update the following constants with your specific environment details:
        ```python
        PROJECT_ID = "your-gcp-project-id"
        LOCATION = "your-gcp-region"  # e.g., "us-central1"
        DATA_STORE_ID = "your-vertex-ai-search-datastore-id"
        DATA_STORE_LOCATION = "global" # Or the specific location of your data store
        ```

4.  **Execution**:
    -   Run the cells in the Jupyter Notebook to initialize the LLM and retriever.
    -   Use the various LangChain chains (`RetrievalQA`, `ConversationalRetrievalChain`, etc.) to ask questions about your documents and see the results.

---

## Example Use Cases

This notebook demonstrates several powerful Q&A capabilities:

1.  **Direct Question Answering**:
    -   **Query**: `"What was Alphabet's Revenue in Q2 2021?"`
    -   **Result**: `"Alphabet's Revenue in Q2 2021 was $61.9 billion."`

2.  **Conversational Follow-up**:
    -   **Initial Query**: `"What were alphabet revenues in 2022?"`
    -   **Follow-up Query**: `"What about costs and expenses?"`
    -   **Result**: The model understands the context and provides the costs and expenses for 2022 without needing the year to be specified again.

3.  **Custom Prompt Engineering**:
    -   **Constraint**: Modify the prompt to force the model to answer with only a single word.
    -   **Query**: `"Were 2020 EMEA revenues higher than 2020 APAC revenues?"`
    -   **Result**: `"Yes"`

