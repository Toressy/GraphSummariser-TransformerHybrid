# 🧠 GraphSummariser-TransformerHybrid

This project implements a **graph-based extractive document summarisation system enhanced with transformer models**. It fuses the structural power of PageRank and HITS with semantic-rich Sentence-BERT (SBERT) embeddings to generate concise, coherent summaries from long-form text.

## 📚 Key Features

- SBERT embeddings for contextual sentence representation
- Graph construction using cosine similarity or cross-encoder similarity
- Ranking via PageRank and HITS algorithms
- Redundancy filtering using cosine similarity
- Comparison with traditional classifiers (SVM, Naïve Bayes, KNN)
- Evaluated on benchmark datasets with ROUGE and BERTScore

## 🧪 Datasets Used

- **BBC News Summary Dataset**
- **CNN/DailyMail News Dataset**

Both datasets are widely used benchmarks in NLP summarisation tasks and are accessible via Hugging Face or Kaggle.

## 🔧 Techniques & Tools

- **SBERT** (`all-MiniLM-L6-v2`)
- **Similarity**: Cosine Similarity & Cross-Encoder (`stsb-distilroberta-base`)
- **Graph Algorithms**: PageRank, HITS (via NetworkX)
- **Redundancy Filter**: Cosine similarity threshold (0.7)
- **Traditional Classifiers**: SVM, KNN, Naïve Bayes
- **Evaluation**: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore

## 🧱 System Architecture


Text → Preprocessing → Sentence Embedding → Similarity Matrix
→ Graph Construction → Sentence Ranking (PageRank/HITS)
→ Redundancy Filtering → Final Summary

## 🧮 Example Methods

  **GCos-PRSum**: Graph + Cosine Similarity + PageRank
  **GTr-HSum**: Graph + Cross-Encoder + HITS
  **SVMSum, NBSum, KNNSum**: Classifier-based baselines
  
## 📈 Evaluation Highlights

### BBC Dataset
  **Best ROUGE-1**: 0.658 (Cross-Encoder + PageRank)
  **Best ROUGE-2**: 0.558 (Graph-based methods)
  **Best BERT-F1**: 0.892 (All transformer graph models)
  
### CNN/DailyMail Dataset
  Slightly lower scores due to document length/complexity
  BERT-F1 consistently strong (~0.783)
  
## 📂 Project Structure

├── graph_rank.py                     # Main implementation script
├── 100661485.Dissertation.docx      # Full dissertation and analysis
└── README.md                        # Project documentation

## 🖥️ Requirements

Python 3.8+
transformers
sentence-transformers
scikit-learn
numpy, pandas
networkx
bert-score
rouge-score
tqdm

### Install with:
pip install -r requirements.txt

## ⚙️ Usage

### Run summarisation:
    python graph_rank.py --input "input_article.txt" --method "pagerank" --embedding "sbert" --summary_size 5

### Available options:
    --method: pagerank, hits, svm, knn, naive_bayes
    --embedding: sbert, crossencoder
    --summary_size: Number of sentences in summary
