import json
import re
import urllib.request
from litellm import completion
import fitz
import numpy as np
import tensorflow_hub as hub
from sklearn.neighbors import NearestNeighbors
from config import openAI_key
import gradio as gr


recommender = None

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    """
    Preprocesses the input text by replacing newline characters with spaces
    and reducing multiple spaces to a single space.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text with newline characters replaced by spaces
             and multiple spaces reduced to a single space.
    """
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def pdf_to_text(path, start_page=1, end_page=None):
    """
    Extracts and preprocesses text from a PDF file within a specified page range.

    Args:
        path (str): The file path to the PDF document.
        start_page (int, optional): The starting page number for text extraction (1-based index). Defaults to 1.
        end_page (int, optional): The ending page number for text extraction (1-based index). If None, extracts until the last page. Defaults to None.

    Returns:
        list: A list of preprocessed text strings, each representing the text content of a page within the specified range.
    """
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list

def text_to_chunks(texts, word_length=150, start_page=1):
    """
    Splits the input texts into chunks of a specified word length and annotates each chunk with its page number.

    Args:
        texts (list of str): A list of text strings, each representing the content of a page.
        word_length (int, optional): The number of words per chunk. Defaults to 150.
        start_page (int, optional): The starting page number for annotation. Defaults to 1.

    Returns:
        list of str: A list of text chunks, each annotated with its corresponding page number.
    """
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    def __init__(self):
        self.use = hub.load(r'C:\Users\u393845\wns\GenAI\universal-sentence-encoder_4')
        self.fitted = False

    def fit(self, data, batch=1000, n_neighbors=5):
        """
        Fits the semantic search model on the provided data by computing embeddings and training a nearest neighbors model.

        Args:
            data (list of str): The input data to fit the model on.
            batch (int, optional): The batch size for processing the data into embeddings. Defaults to 1000.
            n_neighbors (int, optional): The number of neighbors to use for the nearest neighbors model. Defaults to 5.

        Returns:
            None
        """
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=1000):
        """
        Computes embeddings for a list of text strings using a specified batch size.

        Args:
            texts (list of str): The input text strings to compute embeddings for.
            batch (int, optional): The batch size for processing the texts. Defaults to 1000.

        Returns:
            numpy.ndarray: A 2D array where each row represents the embedding of a text string.
        """
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i : (i + batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


def load_recommender(path, start_page=1):
    """
    Loads and fits the recommender model with text data extracted from a PDF file.

    Args:
        path (str): The file path to the PDF document.
        start_page (int, optional): The starting page number for text extraction (1-based index). Defaults to 1.

    Returns:
        str: A message indicating that the corpus has been loaded.
    """
    global recommender
    if recommender is None:
        recommender = SemanticSearch()

    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'

def generate_text(prompt, engine="gpt-4o-mini"):
    try:
        messages=[{ "content": prompt,"role": "user"}]
        completions = completion(
            model=engine,
            messages=messages,
            max_tokens=512,
            n=1,
            stop=None,
            temperature=0.7,
            api_key=openAI_key
        )
        message = completions['choices'][0]['message']['content']
    except Exception as e:
        message = f'API Error: {str(e)}'
    return message


def generate_answer(question):
    """
    Generates an answer to a given question by querying the recommender model and formatting the results into a prompt.

    Args:
        question (str): The question to generate an answer for.

    Returns:
        str: The generated answer based on the search results from the recommender model.
    """
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += (
        "Instructions: Compose a comprehensive reply to the query using the search results given. "
        "Cite each reference using [Page Number] notation (every result has this number at the beginning). "
        "Citation should be done at the end of each sentence. If the search results mention multiple subjects "
        "with the same name, create separate answers for each. Only include information found in the results and "
        "don't add any additional information. Make sure the answer is correct and don't output false content. "
        "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier "
        "search results which has nothing to do with the question. Only answer what is asked. The "
        "answer should be short and concise. Answer step-by-step."
    )

    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(prompt, "gpt-4o-mini")
    return answer


input_file_name = 'handbook.pdf'
questions = ['What is the name of the organization?',
             'Who is the CEO of the company?',
             'What is their vacation policy?',
             'What is the termination policy?']

load_recommender(input_file_name)

results = {}
for question in questions:
    results[question] = generate_answer(question)

# Convert the results dictionary to a JSON string
json_output = json.dumps(results, indent=4)
print(json_output)



# # Create the Gradio interface for demo purpose
#
# def handle_question(uploaded_file, question):
#     load_recommender(uploaded_file.name)
#     answer = generate_answer(question)
#     return answer
#
# def gradio_interface(uploaded_file, question):
#     return handle_question(uploaded_file, question)
#
#
# iface = gr.Interface(
#     fn=gradio_interface,
#     inputs=[
#         gr.File(label="Upload PDF", file_types=[".pdf"]),
#         gr.Textbox(label="Question", placeholder="Ask a question related to the PDF..."),
#     ],
#     outputs= [gr.Textbox(label="Answer")],
#
#     title="PDF Question-Answer Chatbot",
#     description="Upload a PDF, then ask questions about the content."
# )
#
# if __name__ == "__main__":
#     iface.launch()
