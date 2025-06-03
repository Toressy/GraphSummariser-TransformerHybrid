from flask import Flask, render_template, request, jsonify
from graph_rank import summarize_article
import time

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_content = request.form.get('text_content', '')
        methods = request.form.getlist('methods')
        summary_size = int(request.form.get('summary_size', 5))
        
        results = {}
        for method in methods:
            try:
                start_time = time.time()
                
                if method == "graph_cosine_pagerank":
                    summary = summarize_article(
                        text_content,
                        method='graph',
                        similarity_method='cosine',
                        ranking_method='pagerank',
                        summary_size=summary_size
                    )
                elif method == "graph_cosine_hits":
                    summary = summarize_article(
                        text_content,
                        method='graph',
                        similarity_method='cosine',
                        ranking_method='hits',
                        summary_size=summary_size
                    )
                elif method == "graph_transformer_pagerank":
                    summary = summarize_article(
                        text_content,
                        method='graph',
                        similarity_method='transformer',
                        ranking_method='pagerank',
                        summary_size=summary_size
                    )
                elif method == "graph_transformer_hits":
                    summary = summarize_article(
                        text_content,
                        method='graph',
                        similarity_method='transformer',
                        ranking_method='hits',
                        summary_size=summary_size
                    )
                    
                elif method == "svm":
                    summary = summarize_article(
                        text_content,
                        method='svm',
                        summary_size=summary_size
                    )
                elif method == "knn":
                    summary = summarize_article(
                        text_content,
                        method='knn',
                        summary_size=summary_size
                    )
                elif method == "nb":
                    summary = summarize_article(
                        text_content,
                        method='naive_bayes',
                        summary_size=summary_size
                    )
                # Add other methods similarly...
                
                processing_time = time.time() - start_time
                results[method] = {
                    'summary': summary,
                    'time': f"{processing_time:.2f}s"
                }
                
            except Exception as e:
                results[method] = {'error': str(e)}
        
        return jsonify(results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)