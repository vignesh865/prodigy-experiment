#!/usr/bin/env python
# coding: utf-8

import logging

import numpy as np
import plotly.express as px
import qdrant_client
import tiktoken
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
# from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PDFMinerLoader
from langchain.document_transformers import EmbeddingsRedundantFilter
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores import Qdrant
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import English

# In[1]:


DATA_PATH = "/Users/VIGNESH.BASKARAN/Documents/Backups/PersonalProjects/Prodigy/demo-sap-knowledgebase/data/ADM900- 1.pdf"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 10
PDF_COLLECTION_NAME = "sap_security"
OPENAI_API_KEY = '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'


# ### Document Loader

# In[2]:


# In[3]:

def load_data(data_path):
    loader = PDFMinerLoader(DATA_PATH)
    all_pages = loader.load_and_split()

    page_content = ""
    for txt in all_pages[3:]:
        page_content = f"{page_content} {txt.page_content}"

    return [Document(page_content=page_content)]


pages = load_data(DATA_PATH)

# ### Document Transformer

# ### Text embeddings

embeddings = TensorflowHubEmbeddings()
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)


# ### Chunking strategy

# In[8]:


# #### Remove redundant symbols

# In[9]:

def split_with_sentence_transformer(pages):
    final_docs = SentenceTransformersTokenTextSplitter(chunk_overlap=0).split_documents(pages)

    single_page_content = ""

    for i in final_docs:
        single_page_content = f"{single_page_content} {i.page_content}"

    return single_page_content


single_page_content = split_with_sentence_transformer(pages)


# #### Split content into single sentences

# In[10]:

def split_into_sentences(single_page_content):
    nlp = English()

    # Add the component to the pipeline
    nlp.add_pipe('sentencizer')

    #  "nlp" Object is used to create documents with linguistic annotations.
    doc = nlp(single_page_content)

    # create list of sentence tokens
    sents_list = [sent.text for sent in doc.sents]

    print(f"Length of sentence - {len(sents_list)}")
    return sents_list


sents_list = split_into_sentences(single_page_content)


# In[13]:


def duplicated_sentences(sentence_list, repetition_threshold=3):
    sentence_dict = dict(enumerate(sentence_list))
    sentence_vecs = embeddings.embed_documents(sentence_list)
    sentence_vecs_np = np.array([np.array(i) for i in sentence_vecs])

    similarities_matrix = cosine_similarity(sentence_vecs_np,
                                            sentence_vecs_np)

    duplicated_sentences = np.argwhere(similarities_matrix > 0.95)
    duplicated_sentences = np.array([i for i in duplicated_sentences if i[0] != i[1]])

    duplicates_freq = {}

    for i in duplicated_sentences:
        duplicates_freq[i[0]] = duplicates_freq.get(i[0], 0) + 1

    duplicates_above_threshold = []

    for i in duplicated_sentences:
        if duplicates_freq.get(i[0]) > repetition_threshold:
            duplicates_above_threshold.append(i)

    duplicates_above_threshold = np.array(duplicates_above_threshold)

    removed_sentences = []

    for i in set(duplicates_above_threshold[:, 1]):
        removed_sentences.append(sentence_dict.pop(i))

    return list(sentence_dict.values()), removed_sentences


deduplicated_sents, removed_sentences = duplicated_sentences(sents_list)


def get_adjacent_similarities(sentence_vecs_np):
    similarities_matrix = cosine_similarity(sentence_vecs_np, sentence_vecs_np)
    return similarities_matrix.diagonal(1)


def get_adjacent_similarities_from_text(sentence_list):
    sentence_vecs = embeddings.embed_documents(sentence_list)
    sentence_vecs_np = np.array([np.array(i) for i in sentence_vecs])

    return get_adjacent_similarities(sentence_vecs_np)


def plot_similiarites(sentence_list):
    sims = get_adjacent_similarities_from_text(sentence_list)
    fig = px.line(x=range(sims.shape[0]), y=sims, title='Similarities between texts')
    fig.show()


def group_contextually(sentence_list, similarity_threshold, chunk_size):
    sentence_vecs = embeddings.embed_documents(sentence_list)
    sentence_vecs_np = np.array([np.array(i) for i in sentence_vecs])

    adjacent_similarities = get_adjacent_similarities(sentence_vecs_np)

    i = 0

    new_sentence_list = []

    group = sentence_list[i]
    while True:

        current_sentence = sentence_list[i + 1]
        if i + 1 == adjacent_similarities.shape[0]:
            new_sentence_list.append(f"{group} {current_sentence}")
            break

        similarity_diff = abs(adjacent_similarities[i] - adjacent_similarities[i + 1])
        token_len = len(encoding.encode(group)) + len(encoding.encode(current_sentence))

        if similarity_diff > similarity_threshold or token_len > chunk_size:
            new_sentence_list.append(group)
            group = current_sentence

        else:
            group = f"{group} {current_sentence}"

        i = i + 1

    print(current_sentence)
    return new_sentence_list


# In[23]:

def get_chunked_docs(deduplicated_sents):
    new_list = group_contextually(deduplicated_sents, 0.5, CHUNK_SIZE)
    return [Document(page_content=txt) for txt in new_list]


documents = get_chunked_docs(deduplicated_sents)
# In[ ]:


collection_db_ref = {}


# ### Vector store - #client.delete_collection(PDF_COLLECTION_NAME)
def get_vector_db(collection_name):
    if collection_db_ref.get(collection_name) is not None:
        return collection_db_ref.get(collection_name)

    client = qdrant_client.QdrantClient("localhost", port=6333, grpc_port=6333)

    db_ref = Qdrant(
        client=client, collection_name=collection_name,
        embeddings=embeddings,
    )

    collection_db_ref[collection_name] = db_ref

    return db_ref


def insert_vector_representations(documents):
    db_ref = get_vector_db(PDF_COLLECTION_NAME)
    return db_ref.from_documents(documents, embeddings, collection_name=PDF_COLLECTION_NAME)


vector_ref = insert_vector_representations(documents)


# ### Retriever

# #### MultiQueryRetriever
def get_multiquery_retriever(vector_ref):
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    return MultiQueryRetriever.from_llm(
        retriever=vector_ref.as_retriever(), llm=llm
    )


def get_ensemble_retriever(base_llm_retriever, documents):
    txts = [i.page_content for i in documents]
    bm25_retriever = BM25Retriever.from_texts(txts)
    bm25_retriever.k = 2
    return EnsembleRetriever(retrievers=[bm25_retriever, base_llm_retriever], weights=[0.5, 0.5])



def compression_retriever(base_retriever):
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter, redundant_filter]
    )
    return ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=base_retriever)


def get_retriever(vector_ref, documents):
    multiquery_retriever = get_multiquery_retriever(vector_ref)
    ensemble_retriever = get_ensemble_retriever(multiquery_retriever, documents)
    return compression_retriever(ensemble_retriever)


final_retriever = get_retriever(vector_ref, documents)


def get_chain(retriever):
    template = """Use the following pieces of context to answer the question at the end. 
    If yoau don't know the answer, just say that you don't know, don't try to make up an answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        #     chain_type="map_reduce",
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, }
    )


def answer_with_chain(question):
    chain = get_chain(final_retriever)
    return chain({"query": question})








#
# Q2 = "How roles are created in cua"
# qa_chain({"query": Q2})
#
# # In[50]:
#
#
# Q3 = "Which system should be used as a central local for cua?"
# qa_chain({"query": Q3})
#
# # In[51]:
#
#
# Q4 = "How can you find out whether your system is cua configured ?"
# qa_chain({"query": Q4})
#
# # In[52]:
#
#
# Q4 = "How can you find out whether your system is cua configured ?"
# qa_chain({"query": Q4})
#
# # In[88]:
#
#
# Q4 = "what are the required rfc connections for cua?"
# qa_chain({"query": Q4})
#
# # In[54]:
#
#
# Q4 = "How to implement CUA?"
# qa_chain({"query": Q4})
#
# # In[ ]:
#
#
# # In[87]:
#
#
# Q4 = "What are the prerequisites for efficient cua implementation?"
# qa_chain({"query": Q4})
#
# # In[91]:
#
#
# Q4 = "what are the steps to perform RFA connections in transaction?"
# result = qa_chain({"query": Q4})
# print(result['result'])
#
# # In[92]:
#
#
# Q4 = "How to define RFC connections in the transaction SM59?"
# result = qa_chain({"query": Q4})
# print(result['result'])
#
# # In[95]:
#
#
# Q4 = "What are the steps to perform CUA activation?"
# result = qa_chain({"query": Q4})
# print(result['result'])

# In[96]:


