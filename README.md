# PDF-based-questions-and-answers


To further improve performance and scalability we can try these things:

1. Optimize Text Chunking
Current State: The text_to_chunks method chunks text by a fixed word length.
Improvement: Use sentence embeddings to allow a dynamic chunking size based on text similarity or semantic completeness.
Use Sentence Transformers or similar tools to group semantically related sentences into chunks.

2. Improve Embedding Efficiency
Current State: Universal Sentence Encoder (USE) is loaded from a local path, and embedding computation can be slow.
Improvement: Switch to a more modern embedding model like Sentence Transformers (sentence-transformers library) for faster computation and better quality embeddings.

3. Enhance Nearest Neighbor Search
Current State: You use NearestNeighbors from scikit-learn, which may not scale well for large datasets.
Improvement: Use FAISS (Facebook AI Similarity Search) for faster and more efficient nearest-neighbor searches.
Add support for approximate nearest neighbor (ANN) search to handle larger corpora.

4. Optimize the Prompt
Current State: The prompt is well-crafted but static.
Improvement:
Dynamically adjust the prompt length by truncating irrelevant parts of the top chunks if the token limit is exceeded.
  
5. Enhance Preprocessing
Current State: Text preprocessing removes unnecessary spaces and line breaks.
Improvement: Handle OCR-generated PDFs by detecting and fixing broken words or sentences.
Use advanced cleaning tools to normalize fonts, remove headers/footers, or extract tables if needed.
