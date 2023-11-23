import streamlit as st
import os
import openai
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import RawVectorQuery
from azure.search.documents.models import RawVectorQuery



#print("Compiler is here...")
from azure.core.credentials import AzureKeyCredential


OPENAI_API_KEY = "1878732e0a5d4bc6987a8173d820c13d"
OPENAI_API_VERSION = "2023-09-01-preview"
OPENAI_API_ENDPOINT = "https://ibcazureopenai.openai.azure.com/"
OPENAI_API_VERSION = "2023-09-01-preview"

AZURE_COGNITIVE_SEARCH_SERVICE_NAME = "cognsearch2  "
AZURE_COGNITIVE_SEARCH_INDEX_NAME = "operations_index"
AZURE_COGNITIVE_SEARCH_API_KEY = "Z2awepIhuFBevuEic9h0IOSqaXHEVzRK97AnqUIB0XAzSeAYsp9v"
AZURE_COGNITIVE_SEARCH_ENDPOINT = "https://cognsearch2.search.windows.net"
azure_credential = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)


system_message = """Assistant helps the people by answering question asked about IBC bank policies, procedures etc.  
                  Be brief in your answers.\nAnswer ONLY with the facts listed in the list of sources below. 
                  If there isn't enough information below, say you don't know. 
                  Do not generate answers that don't use the sources below.
                  It's very important you follow next step as per instruction being provided. Each source has a name followed by colon and the actual information, always include the source names after your response. To reference the source you should always include atleast 2 new line character and then refrence the source in square brackets. For Example if the source is info.txt then You have to follow this pattern : "\n\n [ Source : info.txt]" . 
                  """


import os
from openai import AzureOpenAI

client = AzureOpenAI(
  api_key = OPENAI_API_KEY,  
  api_version = OPENAI_API_VERSION,
  azure_endpoint = OPENAI_API_ENDPOINT
)

def generate_embeddings_azure_openai(text = " "):
    response = client.embeddings.create(
        input = text,
        model= "ibcembedding"
    )
    return response.data[0].embedding


def generate_final_response(model= "ibcgpt35",
                                  messages= [],
                                  temperature=0.1,
                                  max_tokens = 512,
                                  stream = False):

    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature = temperature,
                                              max_tokens = max_tokens,
                                              stream= stream)

    return response


class retrive_similiar_docs : 

    def __init__(self,query = " ", retrive_fields = ["actual_content", "metadata"],
                      ):
        if query:
            self.query = query

        self.search_client = SearchClient(AZURE_COGNITIVE_SEARCH_ENDPOINT, AZURE_COGNITIVE_SEARCH_INDEX_NAME, azure_credential)
        self.retrive_fields = retrive_fields
    
    def text_search(self,top = 2):
        results = self.search_client.search(search_text= self.query,
                                select=self.retrive_fields,top=3)
        
        return results
        

    def pure_vector_search(self, k = 2, vector_field = 'vector',query_embedding = []):

        vector_query = RawVectorQuery(vector=query_embedding, k=k, fields=vector_field)

        results = self.search_client.search( search_text=None,  vector_queries= [vector_query],
                                            select=self.retrive_fields)

        return results
        
    def hybrid_search(self,top = 2, k = 2,vector_field = "vector",query_embedding = []):
        
        vector_query = RawVectorQuery(vector=query_embedding, k=k, fields=vector_field)
        results = self.search_client.search(search_text=self.query,  vector_queries= [vector_query],
                                                select=self.retrive_fields,top=3)  

        return results


    
    
import time
start = time.time()


def generate_response(user_query = " ", get_sources = False,
                      search_type = "hybrid",top = 2, k =2,stream = False,max_tokens = 512):

    #print("Generating query for embedding...")
    #embedding_query = get_query_for_embedding(user_query=user_query)

    retrive_docs = retrive_similiar_docs(query = user_query)

    if search_type == "text":
        start = time.time()
        r = retrive_docs.text_search(top =3)

        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["metadata"] + ": " + doc["actual_content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["metadata"])
        similiar_docs = "\n".join(similiar_doc)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
        print("Retrived Docs are :",sources,"\n")

    if search_type == "vector":
        start = time.time()
        vector_of_search_query = generate_embeddings_azure_openai(user_query)
        print("Generated embedding for search query in :", time.time()-start,'seconds.')

        start = time.time()
        r = retrive_docs.pure_vector_search(k=3, query_embedding = vector_of_search_query)

        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["metadata"] + ": " + doc["actual_content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["metadata"])
        similiar_docs = "\n".join(similiar_doc)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
        print("Retrived Docs are :",sources,"\n")


    if search_type == "hybrid":
        start = time.time()
        vector_of_search_query = generate_embeddings_azure_openai(user_query)
        print("Generated embedding for search query in :", time.time()-start,'seconds.')

        start = time.time()
        r = retrive_docs.hybrid_search(top = 3, k=3, query_embedding = vector_of_search_query)

        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["metadata"] + ": " + doc["actual_content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["metadata"])
        similiar_docs = "\n".join(similiar_doc)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
        print("Retrived Docs are :",sources,"\n")


    user_content = user_query + " \nSOURCES:\n" + similiar_docs
    chat_conversations = [{"role" : "system", "content" : system_message}]
    chat_conversations.append({"role": "user", "content": user_content })

    if stream: 
        start = time.time()
        response = generate_final_response(messages = chat_conversations,stream = stream,max_tokens=max_tokens)
    
    else : 
        start = time.time()
        response = generate_final_response(messages = chat_conversations,stream = stream,max_tokens=max_tokens)
        print("Generated final response in:", time.time()-start,'seconds.',"\n")
    
    return response   

    

st.title("IBC Assistant")

optional_promt = st.sidebar.selectbox('Select Prompt',
('Prompt1', 'Prompt2', 'Prompt3'))



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            avatar = "🤖"
        else:
            avatar = "🧑‍💻"
        with st.chat_message(message["role"],avatar = avatar ):
            st.markdown(message["content"])

# User input
if prompt := st.chat_input("How can I help?"):
    # Display user message in chat message container
    st.chat_message("user",avatar = "🧑‍💻").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = generate_response(prompt,stream=True)   

    with st.chat_message("assistant",avatar = "🤖"):
        message_placeholder = st.empty()
        full_response = " "
        # Simulate stream of response with milliseconds delay
        for chunk in response:
            if len(chunk.choices) >0:
                if str(chunk.choices[0].delta.content) != "None": 
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

