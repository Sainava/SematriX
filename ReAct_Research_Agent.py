#!/usr/bin/env python
# coding: utf-8

# # Installation

# Libraries which are being installed are:
# 1. `arxiv:` This library provides a way to interact with the `arXiv API`, allowing you to *search for and retrieve academic papers* from the arXiv preprint server.
# 
# 2. `llama_index:` This library (formerly known as `GPT Index`) helps in building an *index over your data*, allowing for *efficient querying* using natural language. It can be **used with large language models (LLMs) for question answering and other tasks**.
# 
# 3. `llama-index-llms-mistralai:` This is an `extension to llama_index` that enables *integration with Mistral AI*, a **LLM provider**. It enables you to use Mistral's LLMs with llama_index.
# 
# 4. `llama-index-embeddings-mistralai:` Similar to the previous one, this extension *allows you to use Mistral's embedding models with llama_index.*
# 
# **NOTE -** Embeddings are numerical representations of text that are used for similarity search within the index.
# 
# In short, `this line of code is setting up the necessary libraries to work with arXiv data, build an index, and leverage Mistral's LLMs and embedding models for querying and processing information.`

# In[60]:


get_ipython().system('pip install arxiv llama_index llama-index-llms-mistralai llama-index-embeddings-mistralai')


# # Importing libraries

# In[61]:


# Importing libraries for HTTP requests and ArXiv interaction
import requests  # For making HTTP requests to download files and interact with APIs
import arxiv  # For interacting with the ArXiv preprint server to search and retrieve papers

# Importing components from the LlamaIndex library
from llama_index.llms.mistralai import MistralAI as MistralLLM  # Importing Mistral's LLM and renaming it for convenience
from llama_index.embeddings.mistralai import MistralAIEmbedding as MistralEmb  # Importing Mistral's embedding model and renaming it
from llama_index.core import (  # Importing core components for indexing and querying
    VectorStoreIndex,  # Used to create and manage the index of documents
    Document,  # Represents a single document in the index
    StorageContext,  # Handles saving and loading the index to/from disk
    load_index_from_storage,  # Function to load a previously saved index
    Settings,  # Allows for configuring various settings of the LlamaIndex library
)
from llama_index.core.tools import (  # Importing tools for building agents
    FunctionTool,  # Allows wrapping Python functions as tools for the agent
    QueryEngineTool,  # Wraps a query engine as a tool for the agent
)
from llama_index.core.agent import ReActAgent  # Importing the ReAct agent class


# # Obtaining API Token & Instantiating the models

# In[62]:


# Securely obtain the API token
api_token= "pSrD5RTTXSVa0YwP8px2IKXOHf08KB5X"

# Instantiate the language model and the embedding model
model_instance = MistralLLM(api_key=api_token, model='mistral-large-latest')
embed_instance = MistralEmb(model_name="mistral-embed", api_key=api_token)


# ## Step 1 – Define an ArXiv Query Function
# 
# * A function is defined to query ArXiv based on a given topic.
# * It constructs a search query and uses the ArXiv API to return details (title, authors, abstract, links, etc.) for a specified number of recent papers.
# * This function provides the dynamic “fetch” capability in case the topic isn’t already in the local knowledge base.

# In[63]:


from typing import List, Dict, Any  # Import necessary type hints
'''
    Queries ArXiv for papers based on the provided topic.
    Returns a list of dictionaries with paper details.
'''
def query_arxiv_papers(topic_str: str, max_results: int) -> List[Dict[str, Any]]:

    # Construct the search query string
    search_term = f'all:"{topic_str}"'

    # Create an ArXiv Search object with search parameters
    search_query = arxiv.Search(
        query=search_term,  # The search query
        max_results=max_results,  # Maximum number of results to fetch
        sort_by=arxiv.SortCriterion.SubmittedDate,  # Sort by submission date
        sort_order=arxiv.SortOrder.Descending  # Sort in descending order (newest first)
    )

    papers_list = []  # Initialize an empty list to store paper details
    try:
        client = arxiv.Client()  # Create an ArXiv API client
        # Iterate through the search results
        for result in client.results(search_query):
            # Extract paper information and store it in a dictionary
            paper_info = {
                'paper_title': result.title,
                'authors_list': [author.name for author in result.authors] if result.authors else [],
                'abstract_text': result.summary,
                'date_published': result.published,
                'journal_details': result.journal_ref,
                'doi_number': result.doi,
                'primary_category': result.primary_category,
                'all_categories': result.categories,
                'pdf_link': result.pdf_url,
                'arxiv_link': result.entry_id
            }
            papers_list.append(paper_info)  # Add the paper info to the list
    except Exception as error:
        # Handle any errors during the ArXiv query
        # In production, use logging instead of print.
        print(f"Error querying ArXiv for '{topic_str}': {error}")

    return papers_list  # Return the list of paper details


# ## Step 2 – Retrieve Initial Papers from ArXiv
# 
# * By calling the ArXiv query function, you fetch a batch of papers (e.g., on "Language Models") to serve as your initial knowledge base.
# 
# * This set will later be converted into documents and indexed for quick retrieval.
# 
# * This function (`query_arxiv_papers`) directly queries ArXiv to retrieve the latest paper details if they aren’t found in the index.

# In[64]:


# Language Models is a search term while 2 is the number of papers
papers_data = query_arxiv_papers("Language Models", 2)


# ## Step 3 – Create Document Objects from Paper Metadata
# 
# * Each paper’s metadata is formatted into a single text block.
# * These text blocks are then converted into Document objects that will be processed by the embedding model.
# * The purpose here is to prepare the raw paper data for vector indexing.

# In[65]:


from typing import List, Dict, Any  # Import necessary type hints
from llama_index.core import Document  # Import the Document class from LlamaIndex

# Converts a list of paper metadata dictionaries into a list of Document objects.
def create_documents(papers_info: List[Dict[str, Any]]) -> List[Document]:
    documents = []  # Initialize an empty list to store Document objects
    for paper in papers_info:  # Iterate through each paper dictionary in the input list
        try:
            # Format the paper metadata into a single string
            content = (
                f"Title: {paper.get('paper_title', 'N/A')}\n"
                f"Authors: {', '.join(paper.get('authors_list', []))}\n"
                f"Abstract: {paper.get('abstract_text', 'N/A')}\n"
                f"Published: {paper.get('date_published', 'N/A')}\n"
                f"Journal: {paper.get('journal_details', 'N/A')}\n"
                f"DOI: {paper.get('doi_number', 'N/A')}\n"
                f"Primary Category: {paper.get('primary_category', 'N/A')}\n"
                f"Categories: {', '.join(paper.get('all_categories', []))}\n"
                f"PDF Link: {paper.get('pdf_link', 'N/A')}\n"
                f"Arxiv Link: {paper.get('arxiv_link', 'N/A')}\n"
            )
            # Create a Document object with the formatted content and add it to the list
            documents.append(Document(text=content))
        except Exception as error:
            # Handle any errors during document creation
            # In production, replace print with logging.error(...)
            print(f"Error processing paper '{paper.get('paper_title', 'Unknown')}': {error}")
    return documents  # Return the list of Document objects

# Assuming 'papers_data' is a list of paper metadata dictionaries obtained from a previous step
doc_objects = create_documents(papers_data)  # Call the function to create Document objects


# ## Step 4 – Build and Persist the Vector Index
# 
# * Using the embedding model, we convert the documents into vector representations and build a vector index.
# 
# * Settings like 'chunk size and overlap' controls how the text is segmented for better embedding.
# 
# * Persisting the index to disk means you don’t have to re-run this expensive step every time you launch the notebook.

# In[66]:


Settings.chunk_size = 1024
Settings.chunk_overlap = 50

# Build the index from the documents using the embedding model
doc_index = VectorStoreIndex.from_documents(doc_objects, embed_model=embed_instance)

# Persist the index to avoid re-indexing on every run
doc_index.storage_context.persist('doc_index/')

# Reload the index from storage
storage_ctx = StorageContext.from_defaults(persist_dir='doc_index/')
doc_index = load_index_from_storage(storage_ctx, embed_model=embed_instance)


# #### Several important steps are happening:
# 
# 1. **Chunking the Documents**  
#    - `Settings.chunk_size = 1024` and `Settings.chunk_overlap = 50` determine how to split each document into smaller, more manageable segments (“chunks”).  
#    - \[**Why chunk?**\] Large text blocks can be harder to embed semantically in a single shot, and also hamper fine-grained search. By slicing text into 1024-token pieces with some overlap between consecutive chunks, you:
#      - Avoid exceeding token or model limits.  
#      - Preserve continuity, because a 50-token overlap means each chunk still has some context from the end of the previous chunk. This helps keep meaning consistent at chunk boundaries.
# 
# 2. **Creating a Vector Index**  
#    - `VectorStoreIndex.from_documents(doc_objects, embed_model=embed_instance)` applies the specified embedding model (`embed_instance`) to each chunk of text. Essentially:
#      - For every chunk of your documents, the model creates a numerical vector (a list of floating-point numbers) that represents the semantic meaning of that text.  
#      - Those vectors are then stored in a specialized data structure (the “vector index”), which allows for efficient similarity search.  
#    - \[**Why embed documents into vectors?**\] Searching with embeddings (a “vector search” approach) is more powerful than keyword search alone, because it lets you find content that’s semantically similar even if it doesn’t match exact keywords.
# 
# 3. **Persisting the Index**  
#    - `doc_index.storage_context.persist('doc_index/')` saves this entire index (the vectors, metadata, structure, etc.) to disk in the `doc_index/` folder.  
#    - This step is important because generating vectors for all your documents can be computational and time intensive. By persisting (storing) it, you don’t need to re-run the embedding process every time you restart or re-run your notebook.  
# 
# 4. **Loading the Index**  
#    - Later (or on the next notebook session), you can do:
#      ```python
#      storage_ctx = StorageContext.from_defaults(persist_dir='doc_index/')
#      doc_index = load_index_from_storage(storage_ctx, embed_model=embed_instance)
#      ```
#      to **reload** the same index from disk.  
#    - This means your system immediately has the previously-created vector data available. You don’t have to re-embed all documents again, saving a lot of time and API calls.
# 
# ---
# 
# ### Why We Need This Step
# 
# 1. **Efficient, Semantic Search**: A vector index lets you retrieve relevant chunks by semantic similarity, rather than simple keyword matching.  
# 
# 2. **Chunking for Model Constraints**: Large documents might exceed the model’s token limit if processed in one go, so chunking is both a technical necessity (token limits) and a best practice (finer-grained search).  
# 
# 3. **Performance & Cost Optimization**: Persisting the index means you pay for embeddings only once. You won’t have to re-embed the same text every time you run the code—this can save significant time and money if you’re using a paid API.  
# 
# In short, this block of code is the backbone of any retrieval-augmented system (RAG) : it turns raw text documents into a searchable vector representation and persists that representation so you can re-use it without repeating expensive operations.

# ## Step 5 – Configure the RAG Query Engine Tool
# 
# • The persisted index is wrapped in a query engine that performs similarity searches over the documents.
# • This query engine is then encapsulated as a “tool” that the agent can use to quickly retrieve information from the static knowledge base.

# In[67]:


search_engine = doc_index.as_query_engine(llm=model_instance, similarity_top_k=5)

rag_tool_instance = QueryEngineTool.from_defaults(
    search_engine,
    name="paper_query_engine",
    description="Query engine using locally indexed research papers."
)


# ### Let's see what’s going on in the above cell
# 
# ### **1. Wrapping the Vector Index into a “Query Engine”**
# 
# - **`doc_index.as_query_engine(llm=model_instance, similarity_top_k=5)`**  
# 
#   - We already have a `doc_index` (the vector index). By calling `.as_query_engine()`, we transform it into a **RAG (Retrieval-Augmented Generation) style** query engine.  
# 
#   That means, When you see `.as_query_engine()`, it transforms the vector index + LLM combination into a single object that:
# 
# - Takes a question,  
# - Does an embedding-based similarity search against the indexed documents,  
# - Passes the retrieved results to your chosen LLM,  
# - Produces a combined, context-rich answer.
# 
#   - **`llm=model_instance`**: You specify which language model to use for summarizing or synthesizing the retrieved chunks. Essentially, when a user query comes in:
#     1. The query is turned into an embedding and compared against the stored vectors in `doc_index`.  
#     2. The top `k` most similar chunks (`similarity_top_k=5`) are pulled up from the index.  
#     3. These retrieved chunks, along with the user’s question, are fed to the language model (`model_instance`) for a final answer or summary.  
# 
# - **Why “query engine?”**  
#   - A query engine orchestrates the whole process of **1) embedding the user query**, **2) performing similarity search**, and **3) optionally letting the LLM summarize** or refine the retrieved chunks into a coherent response. This single object (the “query engine”) is easier to call than manually writing all these steps.
# 
# ---
# 
# ### **2. Creating a Tool for the Agent**
# 
# - **`QueryEngineTool.from_defaults(search_engine, ...)`**  
#   - This line wraps the `search_engine` in a **Tool** that an agent (like a ReAct agent) can call programmatically. Tools are part of the “agent” ecosystem, letting your agent say, “I want to use this particular capability to retrieve knowledge.”  
#   - **`name="paper_query_engine"`** and **`description="Query engine using locally indexed research papers."`** provide a label and a short explanation, so the agent knows what the tool does.  
# 
# - **Why do we need a “Tool?”**  
#   - In many agent frameworks, the agent can choose from multiple specialized Tools (e.g. a “Calculator” tool, a “PDF Downloader” tool, a “Vector Query” tool).  
#   - The ReAct agent or any other sophisticated agent can look at your question, decide it needs to consult the local knowledge base, and “call” this tool with a query. The tool returns relevant chunks or a refined answer to the agent, which the agent then integrates into its final response.
# 
# ---
# 
# ### **Bottom Line**
# 
# 1. **RAG Query Engine**:  
#    The `.as_query_engine()` method transforms your vector index (already containing the embedded chunks of text) into an object that knows how to accept a natural language query, do similarity-based retrieval, and optionally refine the final answer using the language model.
# 
# 2. **Agent Tool**:  
#    Wrapping the query engine as a `QueryEngineTool` means you can embed that retrieval functionality into an agent that can call it like a function. The agent no longer needs to know *how* the search is done under the hood; it just knows there’s a “paper_query_engine” tool available if it wants to look up local research papers.

# ## Step 6 – Define the PDF Download Function
# 
# • A simple function is created to download a PDF file from a provided URL and save it locally.
# • This functionality is critical when you want to keep a copy of a research paper for offline use or further analysis.

# In[68]:


def download_pdf_file(pdf_url, destination):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(destination, "wb") as out_file:
            out_file.write(response.content)
        return f"PDF saved as '{destination}'."
    except Exception as e:
        return f"Error: {e}"


# ## Step 7 – Wrap Functions as Agent Tools
# 
# • Both the ArXiv fetch function and the PDF download function are wrapped as tools.
# 
# • This step lets the agent “call” these functions during a conversation.
# 
# • The static RAG query engine is also wrapped as a tool, so the agent has 3 distinct abilities: search in the index, fetch new papers, and download PDFs.

# In[69]:


pdf_tool = FunctionTool.from_defaults(
    download_pdf_file,
    name="pdf_downloader",
    description="Downloads a PDF given a URL and saves it locally."
)

arxiv_tool = FunctionTool.from_defaults(
    query_arxiv_papers,
    name="arxiv_fetcher",
    description="Fetches recent research papers on a given topic from ArXiv."
)


# ## Step 8 – Integrating Everything with the ReAct Agent
# 
# • Combine the 3 tools—RAG query engine, ArXiv paper fetcher, and PDF downloader—into a single agent using ReAct.
# 
# • The agent first reasons about the query: it checks its knowledge base, and if no relevant papers are found, it will fetch from ArXiv dynamically.
# 
# • The chat flow retains context so that a later command (like “Download the papers…”) refers to the previous result.
# 
# • This orchestration is what allows the agent to combine a fixed knowledge base with dynamic ArXiv queries.

# In[70]:


react_agent = ReActAgent.from_tools(
    [pdf_tool, rag_tool_instance, arxiv_tool],
    llm=model_instance,
    verbose=True
)


# ## Step 9 – Interact with the Agent: Query for Papers
# 
# • A formatted prompt is provided to ask for research papers on a topic (e.g., "Brain-to-Text Decoding").
# 
# • The agent first checks its indexed documents using the RAG tool; if relevant papers are found, it returns their details.
# 
# • If not, it will use the ArXiv fetch tool to get the latest papers.

# In[71]:


prompt_template = (
    "I'm researching {subject}. \n"
    "Using your indexed database, please provide details such as title, abstract, authors, and a link for PDF download for papers related to {subject}. "
    "If nothing is available, fetch the latest ones from ArXiv."
)

response_one = react_agent.chat(prompt_template.format(subject="Brain-to-Text Decoding"))

print(response_one.response)


# ## Step 10 – Instruct the Agent to Download PDFs
# 
# • A follow-up command tells the agent to download the papers it mentioned earlier.
# 
# • This triggers the PDF download tool, which retrieves the PDFs using the provided links and saves them locally.
# 
# ------
# 
# ## So when exactly the agent goes for downloading a paper.
# 
# In the overall project flow, the PDF download isn’t part of the initial query; it’s a follow-up action. Here’s how it works:
# 
# • First, you send a query (using the prompt_template) asking for paper details on a topic (like "Brain-to-Text Decoding").  
#  – The agent checks its local index and, if needed, fetches new papers from ArXiv, returning details (including PDF links).
# 
# • After reviewing the returned paper details, you can issue a new command—for example, “Download the papers you mentioned” or “Save the PDFs for these papers.”  
#  – At this point, the agent uses the conversation context (i.e., the papers it just talked about) and calls the PDF download tool to save the PDFs locally.
# 
# So, you ask for the download after you receive the paper details and decide you want the actual PDF files saved. The project is designed so that the retrieval of paper details and the PDF downloading are separate steps, which gives you control over when you want to perform the download action.
# 
# ------
# 
# ### And ONLY the retrieval of paper details is handled by two components:
# 
# • **RAG Query Engine Tool:**  
# 
#  - This tool searches your pre-indexed (stored) documents and returns details like title, abstract, authors, and PDF link. It’s used when the paper is already in your static knowledge base.
# 
# • **ArXiv Fetch Function (`query_arxiv_papers`):**  
#  - This function directly queries ArXiv to retrieve the latest paper details if they aren’t found in the index.
# 
# Only after these retrieval steps do you later use the PDF download function (wrapped as a tool) when you explicitly instruct the agent to download the PDFs.

# In[72]:


response_two = react_agent.chat("Download the papers you mentioned earlier.")

print(response_two.response)


# ## Step 11 – Test Dynamic ArXiv Fetch for Unindexed Topics
# 
# • Finally, you test the agent with a topic that isn’t in the static index (e.g., “Gaussian process”).
# 
# • The agent uses the ArXiv fetch tool to dynamically retrieve new papers on this topic.
# 
# • This shows the system’s flexibility—combining static retrieval with on-demand querying.
# 
# Finally, query the agent with a topic that isn’t in the index (e.g., “Gaussian process”) to see the ArXiv fetching in action.

# In[73]:


response_three = react_agent.chat(prompt_template.format(subject="Gaussian process"))

print(response_three.response)

