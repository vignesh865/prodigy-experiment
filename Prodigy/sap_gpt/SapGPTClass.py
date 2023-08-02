#!/usr/bin/env python
# coding: utf-8

import logging

import numpy as np
#import plotly.express as px
import qdrant_client
import tiktoken
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
# from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PDFMinerLoader
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.embeddings import GPT4AllEmbeddings
# from langchain.embeddings import TensorflowHubEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores import Qdrant
from spacy.lang.en import English


# In[1]:


class SapGpt:
    # DATA_PATH = "/Users/VIGNESH.BASKARAN/Documents/Backups/PersonalProjects/Prodigy/demo-sap-knowledgebase/data/ADM900- 1.pdf"
    # DATA_PATH = "/Users/VIGNESH.BASKARAN/Downloads/Personal/Vignesh's Resume V9 Purple.pdf"
    DATA_PATH = "Prodigy/demo-files/Vignesh's Resume V9 Purple.pdf"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 10
    # PDF_COLLECTION_NAME = "sap_security"
    PDF_COLLECTION_NAME = "resume"
    OPENAI_API_KEY = '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'

    # ### Document Loader
    def load_data(self, data_path):
        loader = PDFMinerLoader(data_path)
        all_pages = loader.load_and_split()

        page_content = ""
        # for txt in all_pages[3:]:
        for txt in all_pages:
            page_content = f"{page_content} {txt.page_content}"

        return [Document(page_content=page_content)]

    # ### Document Transformer

    # ### Text embeddings

    # ### Chunking strategy

    # In[8]:

    # #### Remove redundant symbols

    # In[9]:

    def split_with_sentence_transformer(self, pages):

        transformer = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
        final_docs = transformer.split_documents(pages)

        single_page_content = ""

        for i in final_docs:
            single_page_content = f"{single_page_content} {i.page_content}"

        return single_page_content

    # #### Split content into single sentences

    def split_into_sentences(self, single_page_content):
        nlp = English()

        # Add the component to the pipeline
        nlp.add_pipe('sentencizer')

        #  "nlp" Object is used to create documents with linguistic annotations.
        doc = nlp(single_page_content)

        # create list of sentence tokens
        sents_list = [sent.text for sent in doc.sents]

        print(f"Length of sentence - {len(sents_list)}")
        return sents_list

    def duplicated_sentences(self, sentence_list, repetition_threshold=3):

        from sklearn.metrics.pairwise import cosine_similarity

        sentence_dict = dict(enumerate(sentence_list))
        sentence_vecs = self.embeddings.embed_documents(sentence_list)
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

        if len(duplicates_above_threshold) == 0:
            return list(sentence_dict.values()), []

        duplicates_above_threshold = np.array(duplicates_above_threshold)

        removed_sentences = []

        for i in set(duplicates_above_threshold[:, 1]):
            removed_sentences.append(sentence_dict.pop(i))

        return list(sentence_dict.values()), removed_sentences

    def get_adjacent_similarities(self, sentence_vecs_np):

        from sklearn.metrics.pairwise import cosine_similarity

        similarities_matrix = cosine_similarity(sentence_vecs_np, sentence_vecs_np)
        return similarities_matrix.diagonal(1)

    def get_adjacent_similarities_from_text(self, sentence_list):
        sentence_vecs = self.embeddings.embed_documents(sentence_list)
        sentence_vecs_np = np.array([np.array(i) for i in sentence_vecs])

        return self.get_adjacent_similarities(sentence_vecs_np)

    # def plot_similiarites(self, sentence_list):
    #     sims = self.get_adjacent_similarities_from_text(sentence_list)
    #     fig = px.line(x=range(sims.shape[0]), y=sims, title='Similarities between texts')
    #     fig.show()

    def group_contextually(self, sentence_list, similarity_threshold, chunk_size):
        sentence_vecs = self.embeddings.embed_documents(sentence_list)
        sentence_vecs_np = np.array([np.array(i) for i in sentence_vecs])

        adjacent_similarities = self.get_adjacent_similarities(sentence_vecs_np)

        i = 0

        new_sentence_list = []

        group = sentence_list[i]
        while True:

            current_sentence = sentence_list[i + 1]
            if i + 1 == adjacent_similarities.shape[0]:
                new_sentence_list.append(f"{group} {current_sentence}")
                break

            similarity_diff = abs(adjacent_similarities[i] - adjacent_similarities[i + 1])
            token_len = len(self.encoding.encode(group)) + len(self.encoding.encode(current_sentence))

            if similarity_diff > similarity_threshold or token_len > chunk_size:
                new_sentence_list.append(group)
                group = current_sentence

            else:
                group = f"{group} {current_sentence}"

            i = i + 1

        print(current_sentence)
        return new_sentence_list

    def get_chunked_docs(self, deduplicated_sents):
        new_list = self.group_contextually(deduplicated_sents, 0.5, SapGpt.CHUNK_SIZE)
        print("Total chunks - ", len(new_list))
        return [Document(page_content=txt) for txt in new_list]

    # ### Vector store - #client.delete_collection(PDF_COLLECTION_NAME)
    def get_vector_db(self, collection_name):
        if self.collection_db_ref.get(collection_name) is not None:
            return self.collection_db_ref.get(collection_name)

        # client = qdrant_client.QdrantClient("localhost", port=6333, grpc_port=6333)
        #client = qdrant_client.QdrantClient(location = ":memory:")
        client = qdrant_client.QdrantClient(
            url="https://e70a4c4d-32b8-491d-a47f-b85967ea8feb.us-east-1-0.aws.cloud.qdrant.io:6333", 
            api_key="B7W75ejwJNKcoLM5E0DkaMZRFyjlC0nNq4BFDmwMQOQhmvZkz4BQng",
        )

        client.delete_collection(collection_name)
        db_ref = Qdrant(
            client=client, collection_name=collection_name,
            embeddings=self.embeddings,
        )

        self.collection_db_ref[collection_name] = db_ref

        return db_ref

    def insert_vector_representations(self, documents):
        db_ref = self.get_vector_db(SapGpt.PDF_COLLECTION_NAME)
        return db_ref.from_documents(documents, self.embeddings, collection_name=SapGpt.PDF_COLLECTION_NAME)

    # ### Retriever

    # #### MultiQueryRetriever
    def get_multiquery_retriever(self, vector_ref):
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        return MultiQueryRetriever.from_llm(
            retriever=vector_ref.as_retriever(), llm=self.llm
        )

    def get_ensemble_retriever(self, base_llm_retriever, documents):
        txts = [i.page_content for i in documents]
        bm25_retriever = BM25Retriever.from_texts(txts)
        bm25_retriever.k = 2
        return EnsembleRetriever(retrievers=[bm25_retriever, base_llm_retriever], weights=[0.5, 0.5])

    def get_compression_retriever(self, base_retriever):
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.76)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[redundant_filter, redundant_filter]
        )
        return ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=base_retriever)

    def get_retriever(self, vector_ref, documents):
        multiquery_retriever = self.get_multiquery_retriever(vector_ref)
        ensemble_retriever = self.get_ensemble_retriever(multiquery_retriever, documents)
        return self.get_compression_retriever(ensemble_retriever)

    def prepare_chain(self, retriever):
        template = """Use the following pieces of context to answer the question at the end. 
        If yoau don't know the answer, just say that you don't know, don't try to make up an answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        return RetrievalQA.from_chain_type(
            self.llm,
            retriever=retriever,
            #     chain_type="map_reduce",
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, }
        )

    def init_prodigy(self):
        pages = self.load_data(SapGpt.DATA_PATH)

        single_page_content = self.split_with_sentence_transformer(pages)
        sents_list = self.split_into_sentences(single_page_content)
        deduplicated_sents, removed_sentences = self.duplicated_sentences(sents_list)
        documents = self.get_chunked_docs(deduplicated_sents)

        vector_ref = self.insert_vector_representations(documents)
        retriever = self.get_retriever(vector_ref, documents)

        self.chain = self.prepare_chain(retriever)

        return self

    def answer_with_chain(self, question):

        if self.chain is None:
            raise ValueError("Chain is not instantiated")

        return self.chain({"query": question.lower()})

    __instance = None

    @staticmethod
    def get_instance(init_prodigy=False):
        """ Static access method. """
        if SapGpt.__instance is None:
            SapGpt(init_prodigy)

        return SapGpt.__instance

    def __init__(self, init_prodigy=False):
        """ Virtually private constructor. """
        if SapGpt.__instance is not None:
            raise Exception("This class is a singleton!")
        else:

            # self.embeddings = TensorflowHubEmbeddings()
            self.embeddings = SentenceTransformerEmbeddings()
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.llm = ChatOpenAI(temperature=0, openai_api_key=SapGpt.OPENAI_API_KEY)
            self.collection_db_ref = {}
            self.chain = None

            if init_prodigy:
                self.init_prodigy()

            SapGpt.__instance = self


if __name__ == '__main__':
    gpt = SapGpt().init_prodigy()
    print("success")

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
