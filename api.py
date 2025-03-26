from flask import Flask, request, jsonify
from flask_cors import CORS
import arxiv

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow API calls from any origin

def query_arxiv_papers(topic_str, max_results):
    search_term = f'all:"{topic_str}"'
    search_query = arxiv.Search(
        query=search_term,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    papers_list = []
    try:
        client = arxiv.Client()
        print(f"Querying ArXiv with search term: {search_term}")  # Debugging log

        for result in client.results(search_query):
            paper_info = {
                "paper_title": result.title,
                "authors_list": [author.name for author in result.authors] if result.authors else [],
                "abstract_text": result.summary,
                "date_published": result.published.strftime('%Y-%m-%d'),
                "journal_details": result.journal_ref if result.journal_ref else "Not available",
                "doi_number": result.doi if result.doi else "Not available",
                "primary_category": result.primary_category,
                "all_categories": result.categories,
                "pdf_link": result.pdf_url,
                "arxiv_link": result.entry_id
            }
            papers_list.append(paper_info)

        if not papers_list:
            return {"error": f"No papers found for '{topic_str}'"}

    except Exception as error:
        print(f"Error querying ArXiv: {error}")  # Log error in terminal
        return {"error": f"Error querying ArXiv for '{topic_str}': {error}"}

    return papers_list


@app.route('/query_papers', methods=['POST'])
def query_papers():
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Invalid request, 'query' field is required"}), 400

        topic = data.get("query", "Artificial Intelligence")  # Default topic
        max_results = int(data.get("max_results", 2))  # ✅ Convert to integer

        papers = query_arxiv_papers(topic, max_results)
        return jsonify({"response": papers})

    except Exception as e:
        print(f"Unexpected Error: {str(e)}")  # Debugging log
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Debug mode enabled
