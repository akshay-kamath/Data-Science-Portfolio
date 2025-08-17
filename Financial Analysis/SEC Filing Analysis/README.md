# Automated SEC Filings Analysis with Vertex AI Gemini

## Overview

This project provides a powerful framework for automating the retrieval, parsing and analysis of public company SEC filings (10-K and 10-Q reports). By leveraging the advanced capabilities of Google's Vertex AI Gemini models, including **function calling** and **grounding with Google Search**, it can overcome common challenges like LLM token limits and data accuracy to deliver insightful financial analysis.

It is designed to act as a financial research assistant that can fetch specific sections from massive SEC documents, compare reports across different companies and time periods and synthesize the information to answer complex questions.

---

## Key Features

-   **Automated Filing Retrieval**: Downloads 10-K (annual) and 10-Q (quarterly) reports directly from the SEC EDGAR database for any public company.
-   **Intelligent Section Parsing**: Overcomes LLM token limits by strategically parsing large HTML filings to extract specific, targeted sections (eg "Risk Factors," "Management’s Discussion and Analysis").
-   **Reliable Company CIK Lookup**: Integrates **Google Search (Grounding)** as a tool to accurately find a company's Central Index Key (CIK), preventing model hallucination and ensuring the correct documents are retrieved.
-   **Advanced Tool Integration**: Utilizes Gemini's **function calling** capabilities to empower the model to interact with custom Python functions for data retrieval and processing.
-   **Comparative and Longitudinal Analysis**: Capable of performing complex analyses such as comparing risk factors between competitors or tracking the evolution of a company's business strategy over several years.

---

## How It Works

The workflow is orchestrated through a Python script that interacts with the Vertex AI API:

1.  **CIK Lookup (Grounding)**: When a company name is provided, the system first uses a Gemini model grounded with the **Google Search tool** to find the official 10-digit CIK. This ensures high accuracy for the initial data retrieval step.

2.  **Filing Download**: Using the CIK, a Python function queries the SEC EDGAR API to find and download the relevant filings for a specified date range.

3.  **Section Extraction**: Since full SEC filings often exceed the LLM's context window, the script uses `BeautifulSoup` to parse the document's HTML structure. It intelligently identifies and extracts the text from only the requested sections (eg, "Item 1A. Risk Factors").

4.  **LLM-Powered Analysis (Function Calling)**: The extracted text is then passed to a Gemini model that has been configured with a `Tool` containing our Python functions. The model can autonomously decide to call these functions to fetch the necessary data to answer a user's prompt. For example, when asked to compare two companies, it will perform two separate function calls to retrieve the filings for each.

5.  **Synthesized Response**: Finally, the model uses the retrieved information to generate a comprehensive, data-driven analysis, quoting directly from the source documents to support its conclusions.

---

## Tech Stack

-   **AI/ML**: Google Vertex AI (Gemini 2.5 Flash)
-   **Core Libraries**: Python
-   **Data Retrieval**: `requests` (for SEC EDGAR API)
-   **Web Scraping/Parsing**: `BeautifulSoup`
-   **Key AI Concepts**: Function Calling, Grounding with Google Search

---

## Example Use Cases

This notebook demonstrates several powerful analytical capabilities:

1.  **Competitor Risk Analysis**:
    -   **Prompt**: *"How do Alphabet's risks in 2024 compare to Amazon's?"*
    -   **Action**: The model looks up the CIKs for both companies, downloads their latest 10-K filings, extracts the "Risk Factors" section from each, and provides a detailed comparative analysis.

2.  **Longitudinal Strategy Analysis**:
    -   **Prompt**: *"How has Home Depot changed the way it describes its business over the past 3 years?"*
    -   **Action**: The model downloads Home Depot's 10-K filings for the last three years, extracts the "Business" section from each, and synthesizes the changes in strategy, focus, and market outlook over time.

---

## Setup and Usage

1.  **Prerequisites**:
    -   A Google Cloud Platform (GCP) project with the Vertex AI API enabled.
    -   Authenticated `gcloud` CLI.
2.  **Installation**:
    -   Install the required Python libraries: `google-cloud-aiplatform`, `requests`, `beautifulsoup4`.
3.  **Configuration**:
    -   Initialize the Vertex AI client with your project ID and location.
    -   Set the `PARTNER_COMPANY` and `PARTNER_WEBSITE` variables in the script to comply with SEC EDGAR API usage policies.
4.  **Execution**:
    -   Run the cells in the Jupyter Notebook to perform the analysis. You can modify the prompts in the final cells to ask your own questions.
