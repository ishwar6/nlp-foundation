import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, PodSpec
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore


PINECONE_API_KEY=''
OPENAI_API_KEY=''


import os
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


class DocumentProcessor:
    """
    A class to process documents, generate embeddings, create and query an index for document retrieval.
    """
    
    def __init__(self, pdf_path, openai_api_key=OPENAI_API_KEY, pinecone_api_key=PINECONE_API_KEY):
        """
        Initializes the DocumentProcessor with paths and keys.
        
        :param pdf_path: Path to the PDF document.
        :param openai_api_key: API key for OpenAI services.
        :param pinecone_api_key: API key for Pinecone services.
        """
        self.pdf_path = pdf_path
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.index_name = "querydocs"
        self.dimension = 3072  # Dimension of the embeddings
        self.metric = "cosine"  # Similarity metric for Pinecone

        # Initialize services
        self.loader = PyPDFLoader(self.pdf_path)
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=self.openai_api_key)
        self.pinecone_client = self.initialize_pinecone()

    def initialize_pinecone(self):
        """
        Initializes the Pinecone service and creates an index if it does not exist.
        
        :return: Pinecone client object
        """
        pc = Pinecone(api_key=self.pinecone_api_key)
        # if self.index_name not in pc.list_indexes():
        #     pc.create_index(
        #         name=self.index_name,
        #         dimension=self.dimension,
        #         metric=self.metric,
        #         spec=PodSpec(
        #             environment="us-west-1-gcp",
        #             pod_type="p1.x1",
        #             pods=1
        #         )
        #     )
        return pc

    def chunk_data(self, docs, chunk_size=800, chunk_overlap=50):
        """
        Chunks the document data into smaller parts for processing.
        
        :param docs: Documents to be chunked.
        :param chunk_size: Size of each chunk.
        :param chunk_overlap: Overlap size between chunks.
        :return: List of document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(docs)

    def process_documents(self):
        """
        Processes documents by loading, chunking, generating embeddings, and indexing them.
        
        :return: None
        """
        docs = self.loader.load_and_split()
        # We can connect to our Pinecone index and insert those chunked docs as contents with PineconeVectorStore.from_documents.

        # https://python.langchain.com/docs/integrations/vectorstores/pinecone
        # docsearch = PineconeVectorStore.from_documents(docs, self.embeddings_model, index_name=self.index_name)
        # return docsearch


        # Ensure each page is a string and not None or another type.
        valid_pages = [page for page in docs if isinstance(page, str) and page.strip()]

        # Process each valid page to generate embeddings.
        # Note: Consider implementing batch processing for efficiency.
        documents_embeddings = []
        for page in valid_pages:
            try:
                embedding = self.embeddings_model.embed_query(page)
                documents_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing page: {e}")
                # Optionally, log the error or handle it as needed.

        # Index documents with embeddings.
        # This loop assumes `embed_query` returns embeddings in a format compatible with Pinecone's upsert method.
        for doc_id, embedding in enumerate(documents_embeddings):
            self.pinecone_client.upsert(vectors=[(str(doc_id), embedding)])


    def retrieve_answers(self, query):
        """
        Retrieves answers for a given query by searching the indexed documents and generating a response.
        
        :param query: The query string.
        :return: The response generated by the LLM.
        """
        doc_search = self.retrieve_query(query)
        # print("dco search", doc_search)
        llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.5, api_key=self.openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        # result = chain({"input_documents": doc_search, "human_input": query}, return_only_outputs=True)
        input_data = {
            "question": query,
            "input_documents": doc_search  # Make sure this matches the expected structure
            }

        response = chain.invoke(input=input_data)
        print("response", response)
        return response

    def retrieve_query(self, query, k=2):
        # Implementation example
        # try:
            # Your existing logic to retrieve documents

            index = self.pinecone_client.Index(name = self.index_name)
            print(index)
            index = PineconeVectorStore(index=index, embedding=self.embeddings_model)
            print(index)
            matching_results = index.similarity_search(query, k=k)
            
            print(matching_results)
            return matching_results
        # except Exception as e:
        #     print(f"Error retrieving query results: {e}")
        #     return []  # Ensure an iterable is always returned



pdf_path = "/Users/ishwarjangid/Desktop/querydocs/llm/pdf/python_book_go4.pdf"

doc_processor = DocumentProcessor(pdf_path, OPENAI_API_KEY, PINECONE_API_KEY)
doc_processor.process_documents()
answer = doc_processor.retrieve_answers("what is strategy pattern")
# print(answer)