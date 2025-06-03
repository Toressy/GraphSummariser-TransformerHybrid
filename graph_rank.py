import os
import re
import time
import torch
import numpy as np
import networkx as nx
import nltk
nltk.download('punkt')  # Not 'punkt_tab' - that was a typo in the error
nltk.download('stopwords')  # Optional but recommended
from datetime import datetime
from datasets import load_dataset
from functools import lru_cache, partial
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, CrossEncoder
from bert_score import BERTScorer
from tqdm.auto import tqdm
from collections import defaultdict
import warnings
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler
from huggingface_hub import login

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TOKENIZERS_NO_THREADING"] = "1"
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore")

# Initialize models
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SBERT_MODEL = 'all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL = 'cross-encoder/stsb-distilroberta-base'
BERT_SCORER_MODEL = 'distilbert-base-uncased'

print(f"Initializing models on {DEVICE}...")
try:
    sbert_model = SentenceTransformer(SBERT_MODEL, device=DEVICE)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device='cpu')
    bert_scorer = BERTScorer(lang="en", model_type=BERT_SCORER_MODEL, device=DEVICE)
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print("Using fallback models...")
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=DEVICE)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
    bert_scorer = BERTScorer(lang="en", model_type='microsoft/deberta-base-mnli', device=DEVICE)

# Data loading functions
def load_bbc_dataset(path="BBC_News_Summary_01", max_articles=500):
    """Load BBC News dataset from local files"""
    categories = ["business", "entertainment", "politics", "sport", "tech"]
    data = []
    
    for category in categories:
        articles_path = os.path.join(path, "News Articles", category)
        summaries_path = os.path.join(path, "Summaries", category)
        
        if not os.path.exists(articles_path):
            continue
            
        for file_name in os.listdir(articles_path):
            if len(data) >= max_articles:
                break
                
            if file_name.endswith(".txt"):
                article_path = os.path.join(articles_path, file_name)
                summary_path = os.path.join(summaries_path, file_name)
                
                try:
                    with open(article_path, "r", encoding='utf-8') as f:
                        article_text = f.read().strip()
                    with open(summary_path, "r", encoding='utf-8') as f:
                        summary_text = f.read().strip()
                        
                    data.append({
                        "category": category,
                        "article": article_text,
                        "summary": summary_text
                    })
                except Exception as e:
                    print(f"Error loading {file_name}: {str(e)}")
                    continue
                    
    return data

def load_cnn_dailymail_dataset(split='train[:3]'):
    """Load CNN/DailyMail dataset from HuggingFace"""
    try:
        dataset = load_dataset('cnn_dailymail', '3.0.0', split=split)
        # Standardize the field names
        processed_data = []
        for item in dataset:
            processed_data.append({
                "article": item["article"],
                "summary": item["highlights"]
            })
        return processed_data
    except Exception as e:
        print(f"Error loading CNN/DailyMail dataset: {str(e)}")
        return []

# Preprocessing
@lru_cache(maxsize=5000)
def cached_encode(sentence):
    return sbert_model.encode(sentence, convert_to_tensor=False)

def preprocess_article(article, max_length=2000):
    article = re.sub(r'\s+', ' ', article)[:max_length]
    return sent_tokenize(article)

# Graph-based methods
def calculate_similarity_matrix(sentences, method='cosine'):
    if method == 'cosine':
        embeddings = np.array([cached_encode(s) for s in sentences])
        return cosine_similarity(embeddings)
    elif method == 'transformer':
        pairs = [(i, j) for i in range(len(sentences)) for j in range(i+1, len(sentences))]
        batch_size = min(32, len(pairs))
        
        with ThreadPoolExecutor() as executor:
            process_fn = partial(process_pair, sentences=sentences)
            similarity_scores = list(executor.map(process_fn, pairs, chunksize=batch_size))
        
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for (i, j), score in zip(pairs, similarity_scores):
            similarity_matrix[i][j] = score
            similarity_matrix[j][i] = score
        return similarity_matrix
    
def process_pair(pair, sentences):
    i, j = pair
    return cross_encoder.predict([(sentences[i], sentences[j])], show_progress_bar=False)[0]

def build_and_rank_graph(similarity_matrix, sentences, embeddings=None, summary_size = 5,  method='pagerank') :
    threshold = np.percentile(similarity_matrix, 80)
    similarity_matrix[similarity_matrix < threshold] = 0
    np.fill_diagonal(similarity_matrix, 0)
    
    graph = nx.from_numpy_array(similarity_matrix)
    
    if method == 'pagerank':
        scores = nx.pagerank(graph, weight='weight')
    elif method == 'hits':
        _, scores = nx.hits(graph)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    selected_sentences = remove_redundant_sentences(
            sentences=sentences,
            embeddings=embeddings,
            scores=scores,  # Graph-based scores
            threshold=0.7,
            summary_size=summary_size
        )
        
    return ' '.join(selected_sentences)
    

def remove_redundant_sentences(sentences, embeddings=None, scores=None, threshold=0.7, summary_size=5):
    """Flexible redundancy removal with summary size limit"""
    selected = []
    
    items = [
        (scores[i] if scores is not None else 1, i, sent)
        for i, sent in enumerate(sentences)
    ]
    
    items.sort(reverse=True, key=lambda x: x[0])
    
    for score, idx, sent in items:
        if len(selected) >= summary_size:
            break
            
        if embeddings is not None:
            is_redundant = any(
                cosine_similarity([embeddings[idx]], [embeddings[s[0]]])[0][0] >= threshold
                for s in selected
            )
        else:
            is_redundant = any(
                cosine_similarity([sbert_model.encode(sent)], 
                                [sbert_model.encode(selected_sent)])[0][0] >= threshold
                for _, selected_sent in selected
            )
            
        if not is_redundant:
            selected.append((idx, sent))
            
    return [sent for idx, sent in sorted(selected, key=lambda x: x[0])]

# Classifier-based methods
def summarize_with_classifier(sentences, embeddings, classifier_type='svm', summary_size=5):
    # Prepare features and pseudo-labels
    features = embeddings
    labels = np.zeros(len(sentences))
    labels[:summary_size] = 1  # First N sentences as positive examples
    
    # Train classifier
    if classifier_type == 'svm':
        model = SVC(kernel='linear', probability=True)
    elif classifier_type == 'naive_bayes':
        #scaler = MinMaxScaler()
        #features = scaler.fit_transform(features)  # NB requires non-negative features
        model = BernoulliNB()
    elif classifier_type == "knn":
        n_neighbors = min(9, len(sentences) - 1)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")
    
    model.fit(features, labels)
    
    # Predict probabilities
    probabilities = model.predict_proba(features)[:, 1]
    
    selected_sentences = remove_redundant_sentences(
        sentences=sentences,
        embeddings=embeddings,
        scores=probabilities,  # Classifier scores
        threshold=0.7,  # Slightly lower threshold for ML
        summary_size=summary_size
    )
    
    return ' '.join(selected_sentences)

# Unified summarization function
def summarize_article(article, method='graph', similarity_method='cosine', 
                     ranking_method='pagerank', summary_size=5):
    sentences = preprocess_article(article)
    if len(sentences) < 3:
        return ""
    
    embeddings = np.array([cached_encode(s) for s in sentences])
    
    if method == 'graph':
        similarity_matrix = calculate_similarity_matrix(sentences, method=similarity_method)
        return build_and_rank_graph(similarity_matrix, sentences, embeddings, summary_size, method=ranking_method)
        


    elif method in ['svm', 'naive_bayes', 'knn']:
        return summarize_with_classifier(sentences, embeddings, 
                                       classifier_type=method, 
                                       summary_size=summary_size)
    else:
        raise ValueError(f"Unknown method: {method}")

# Evaluation
def evaluate_batch(dataset, methods_config, dataset_name="Unknown", summary_size=5):
    results = defaultdict(lambda: defaultdict(list))
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    
    for sample in tqdm(dataset, desc="Processing articles"):
        article = sample['article']
        reference = sample['summary']
        
        for config in methods_config:
            method_type = config['method']
            try:
                summary = summarize_article(
                    article,
                    method=method_type,
                    similarity_method=config.get('similarity', 'cosine'),
                    ranking_method=config.get('ranking', 'pagerank'),
                    summary_size=summary_size
                )
                
                # ROUGE scores
                rouge = scorer.score(reference.lower(), summary.lower())
                for key in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
                    results[str(config)][f'rouge_{key}'].append(rouge[key].fmeasure)
                
                # BERTScore
                P, R, F1 = bert_scorer.score([summary], [reference])
                results[str(config)]['bert_p'].append(P.mean().item())
                results[str(config)]['bert_r'].append(R.mean().item())
                results[str(config)]['bert_f1'].append(F1.mean().item())
                
            except Exception as e:
                print(f"Error with {config}: {str(e)}")
                continue
    
    return results

def save_results(results, dataset_name, dataset_len, output_dir="results"):
    """Save evaluation results to file"""
    # Calculate averages
    final_results = {}
    for config, metrics in results.items():
        final_results[config] = {
            'rouge1': np.mean(metrics['rouge_rouge1']),
            'rouge2': np.mean(metrics['rouge_rouge2']),
            'rougeL': np.mean(metrics['rouge_rougeL']),
            'rougeLsum': np.mean(metrics['rouge_rougeLsum']),
            'bert_f1': np.mean(metrics['bert_f1'])
        }

    # Print and save results
    print("\n=== Final Results ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"summary_results_{timestamp}.txt")
    
    with open(filename, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of articles: {dataset_len}\n")
        f.write("\n=== Evaluation Results ===\n\n")
        for config, scores in final_results.items():
            line = (f"{config:<72}: "
                    f"ROUGE-1 = {scores['rouge1']:.3f} | "
                    f"ROUGE-2 = {scores['rouge2']:.3f} | "
                    f"ROUGE-L = {scores['rougeL']:.3f} | "
                    f"ROUGE-Lsum = {scores['rougeLsum']:.3f} | "
                    f"BERT-F1 = {scores['bert_f1']:.3f}")
            
            print(line)
            f.write(line + "\n")
    
    return filename

if __name__ == "__main__":
    start_time = time.time()
    
    # Configuration
    DATASET_CHOICE = "cnn"  # "bbc" or "cnn"
    DATASET_SIZE = 3     
    SUMMARY_SIZE = 3        # 5 for bbc and 3 for cnn
    
    # Load data
    print("Loading dataset...")
    try:
        if DATASET_CHOICE == "bbc":
            data = load_bbc_dataset(max_articles=500)
            dataset = data[:DATASET_SIZE]
            dataset_name = "BBC News Summary"
        elif DATASET_CHOICE == "cnn":
            dataset = load_cnn_dailymail_dataset(split=f'train[:{DATASET_SIZE}]')
            dataset_name = "CNN/DailyMail"
        else:
            raise ValueError("Invalid dataset choice. Use 'bbc' or 'cnn'")
            
        if not dataset:
            raise ValueError("No data loaded")
            
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        exit(1)
    
    # Define evaluation configurations
    methods_config = [
        # Graph-based methods
        {'method': 'graph', 'similarity': 'cosine', 'ranking': 'pagerank'},
        {'method': 'graph', 'similarity': 'cosine', 'ranking': 'hits'},
        {'method': 'graph', 'similarity': 'transformer', 'ranking': 'pagerank'},
        {'method': 'graph', 'similarity': 'transformer', 'ranking': 'hits'},
        
        # Classifier methods
        {'method': 'svm'},
        {'method': 'naive_bayes'},
        {'method': 'knn'},
    ]
    
    # Run evaluation
    try:
        results = evaluate_batch(dataset, methods_config, dataset_name, SUMMARY_SIZE)
        results_file = save_results(results, dataset_name, len(dataset))
        
        print(f"\nResults saved to: {results_file}")
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")