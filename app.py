import streamlit as st
import time
from graph_rank import summarize_article  # Your existing summarization code

# Page configuration
st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for better styling
st.markdown(""" 
    <style>
    .main {
        max-width: 1000px;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stTextArea textarea {
        min-height: 200px;
    }
    .summary-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .header {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# App header
st.title("📄 Document Summarizer")
st.markdown("Upload a text file or paste text to generate summaries using different algorithms.")

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = {}

# Input options
input_method = st.radio("Input method:", ("Upload file", "Paste text"), horizontal=True)

text_content = ""
if input_method == "Upload file":
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    if uploaded_file:
        text_content = uploaded_file.read().decode("utf-8")
else:
    text_content = st.text_area("Paste your text here:", height=200)

# Summarization options
col1, col2 = st.columns(2)
with col1:
    st.subheader("Summarization Methods")
    methods = st.multiselect(
        "Select methods to compare:",
        options=[
            "Graph (Cosine + PageRank)",
            "Graph (Cosine + HITS)",
            "Graph (Transformer + PageRank)",
            "Graph (Transformer + HITS)",
            "SVM Classifier",
            "KNN Classifier",
            "Naive Bayes Classifier"
        ],
        default=["Graph (Cosine + PageRank)", "SVM Classifier"]
    )

with col2:
    st.subheader("Settings")
    summary_size = st.slider("Summary length (sentences):", 3, 10, 5)
    show_original = st.checkbox("Show original text", False)

# Process button
process_btn = st.button("Generate Summaries")

# Results display
if process_btn and text_content:
    with st.spinner("Processing..."):
        # Clear previous results
        st.session_state.results = {}
        
        # Process selected methods
        for method in methods:
            try:
                start_time = time.time()
                
                if method == "Graph (Cosine + PageRank)":
                    summary = summarize_article(
                        text_content,
                        method='graph',
                        similarity_method='cosine',
                        ranking_method='pagerank',
                        summary_size=summary_size
                    )
                elif method == "Graph (Cosine + HITS)":
                    summary = summarize_article(
                        text_content,
                        method='graph',
                        similarity_method='cosine',
                        ranking_method='hits',
                        summary_size=summary_size
                    )
                elif method == "Graph (Transformer + PageRank)":
                    summary = summarize_article(
                        text_content,
                        method='graph',
                        similarity_method='transformer',
                        ranking_method='pagerank',
                        summary_size=summary_size
                    )
                elif method == "Graph (Transformer + HITS)":
                    summary = summarize_article(
                        text_content,
                        method='graph',
                        similarity_method='transformer',
                        ranking_method='hits',
                        summary_size=summary_size
                    )
                    
                elif method == "SVM Classifier":
                    summary = summarize_article(
                        text_content,
                        method='svm',
                        summary_size=summary_size
                    )
                elif method == "KNN Classifier":
                    summary = summarize_article(
                        text_content,
                        method='knn',
                        summary_size=summary_size
                    )
                elif method == "Naive Bayes Classifier":
                    summary = summarize_article(
                        text_content,
                        method='naive_bayes',
                        summary_size=summary_size
                    )
                
                processing_time = time.time() - start_time
                st.session_state.results[method] = {
                    'summary': summary,
                    'time': f"{processing_time:.2f}s"
                }
                
            except Exception as e:
                st.error(f"Error with {method}: {str(e)}")
                continue

    # Display results
    if st.session_state.results:
        st.success("Summarization complete!")
        st.subheader("Results")
        
        # Show original if requested
        if show_original:
            with st.expander("Original Text"):
                st.text(text_content)
        
        # Display summaries
        for method, result in st.session_state.results.items():
            with st.container():
                st.markdown(f"<div class='summary-box'>", unsafe_allow_html=True)
                st.markdown(f"### {method}")
                st.markdown(f"⏱️ Processing time: {result['time']}")
                st.markdown("**Summary:**")
                st.write(result['summary'])
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No summaries generated. Please check your input and try again.")

elif process_btn and not text_content:
    st.warning("Please provide text content to summarize.")

# Instructions
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    1. **Input your text**: Either upload a .txt file or paste text directly
    2. **Select methods**: Choose which summarization algorithms to compare
    3. **Adjust settings**: Control summary length and other options
    4. **Generate summaries**: Click the button to process your document
    
    ### About the Methods:
    - **Graph-based**: Uses sentence relationships to identify important content
    - **Classifier-based**: Machine learning approaches to select key sentences
    - Different methods may work better for different types of documents
    """)

# Footer
st.markdown("---")
st.markdown("Document Summarizer v1.0 | Built with Streamlit")