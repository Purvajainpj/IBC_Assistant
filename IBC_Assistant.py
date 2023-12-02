import streamlit as st
import os
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import RawVectorQuery
from azure.search.documents.models import RawVectorQuery
from azure.core.credentials import AzureKeyCredential

## Function to check login credentials
def authenticate(email, password):
   # Replace with your authentication logic
   # For simplicity, a hardcoded email and password are used
   valid_email = "ibchelpdesk@ibc.com"
   valid_password = "Myfriend@1234"
   return email == valid_email and password == valid_password
# Session state initialization
if 'logged_in' not in st.session_state:
   st.session_state.logged_in = False
# Login Page
login = st.sidebar.checkbox("Login")
if login and not st.session_state.logged_in:
   st.sidebar.title("Login")
   email = st.sidebar.text_input("Email")
   password = st.sidebar.text_input("Password", type="password")
   if st.sidebar.button("Login"):
       if authenticate(email, password):
           st.session_state.logged_in = True
           st.experimental_rerun()
       else:
           st.sidebar.error("Invalid email or password")
# Check if the user is logged in before proceeding
if not st.session_state.logged_in:
   st.warning("Please log in to use the IBC Assistant.")
   st.stop()  # Stop further execution if not logged in

st.title("IBC Assistant")


OPENAI_API_KEY = "b8717ce6243449aab913a9163e939af9"
OPENAI_API_ENDPOINT = "https://ibcazureopenainew.openai.azure.com/"
OPENAI_API_VERSION = "2023-09-01-preview"

AZURE_COGNITIVE_SEARCH_SERVICE_NAME = "cognsearch2  "
AZURE_COGNITIVE_SEARCH_API_KEY = "Z2awepIhuFBevuEic9h0IOSqaXHEVzRK97AnqUIB0XAzSeAYsp9v"
AZURE_COGNITIVE_SEARCH_ENDPOINT = "https://cognsearch2.search.windows.net"
azure_credential = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)

#AZURE_COGNITIVE_SEARCH_INDEX_NAME = "process_all_category_index"
AZURE_COGNITIVE_SEARCH_INDEX_NAME = "final_500_index"




logo_url = "https://www.ibc.com/images/ibc-logo.png"
logo_html = f'<img src="{logo_url}" alt="Logo" height="60" width="200">'
st.sidebar.markdown(f'<div class="logo-container">{logo_html}</div>', unsafe_allow_html=True)




nationality = st.sidebar.selectbox('Select Business',("None",'Domestic', 'Foreign'))

if nationality == "None" :
    dropdown_1_prompt = " "
    
if nationality == "Domestic" :
    dropdown_1_prompt = "you have to answer as the question is being asked in context of a domestic businesses."

if nationality == "Foreign" :
    dropdown_1_prompt = "you have to answer as the question is being asked in context of a foreign businesses."



account_type = st.sidebar.selectbox('Select Individul/Business',("None",'Personal', 'Business'))

if account_type == "None" :
    dropdown_2_prompt = " "
    
if account_type == "Personal" :
    dropdown_2_prompt = "you have to answer as the question is being asked in context of an Personal account."

if account_type == "Business" :
    dropdown_2_prompt = "you have to answer as the question is being asked in context of an Business account."




category = st.sidebar.selectbox('Select Category',("General",'Operations', 'Quality Control',"UCC", "BSA-AML", "BSA-Regs","Consumer Regs", "IT procedures","Free Checking"))


if category == "General":
    #AZURE_COGNITIVE_SEARCH_INDEX_NAME = "free_checking_index"
    
    dropdown_3_prompt = " "

if category == "Operations":
    #AZURE_COGNITIVE_SEARCH_INDEX_NAME = "free_checking_index"
    
    dropdown_3_prompt = "you have to answer as the question is being asked in context of operations category"
    
if category == "Quality Control":
    #AZURE_COGNITIVE_SEARCH_INDEX_NAME = "quality_control_5000_index"
    
    dropdown_3_prompt = "you have to answer as the question is being asked in context of Quality control category."
    
if category == "UCC Regs":
    #AZURE_COGNITIVE_SEARCH_INDEX_NAME = "ucc_regs_index"
    
    dropdown_3_prompt = "you have to answer as the question is being asked in context of UCC Regs category."
    
if category == "BSA-AML":
    #AZURE_COGNITIVE_SEARCH_INDEX_NAME = "bsa_aml_index"
    
    dropdown_3_prompt = "you have to answer as the question is being asked in context of BSA-AML category."
    
if category == "BSA-Regs":
    #AZURE_COGNITIVE_SEARCH_INDEX_NAME = "bsa_reg_index"
    
    dropdown_3_prompt = "you have to answer as the question is being asked in context of BSA-Regs category."
    
if category == "Consumer Regs":
    #AZURE_COGNITIVE_SEARCH_INDEX_NAME = "consumer_reg_index"
    
    dropdown_3_prompt = "you have to answer as the question is being asked in context of Consumer Regs category."
    
if category == "IT procedures":
    #AZURE_COGNITIVE_SEARCH_INDEX_NAME = "it_procedure_index"
    
    dropdown_3_prompt = "you have to answer as the question is being asked in context of IT procedures category."
    
if category == "Free Checking":
    #AZURE_COGNITIVE_SEARCH_INDEX_NAME = "free_checking_index
    
    dropdown_3_prompt = "you have to answer as the question is being asked in context of Free Checking category."
    


system_message = """Assistant provides precise answers based on IBC bank policies and procedures. Be concise in your responses, sticking strictly to the facts from the listed sources below. If information is insufficient, indicate that you don't know.""" + dropdown_1_prompt + dropdown_2_prompt + dropdown_3_prompt + """
Always answer exactly what the user asks, avoiding unnecessary details. Reference sources by including at least 2 new line characters followed by the source in square brackets, like this: "\n\n [ Source : info.txt]". 
Examples:

Question: What percentage should be documented for Beneficial Ownership of an entity?

Answer: The percentage requered for beneficial ownership is 25%.

Question: How long can the bank take to investigate a debit card dispute?

Answer: The bank investigates debit card disputes within 45 days, providing provisional credit within 10 business days of receiving the error notice.

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
        model= "ibcembeddingada2"
    )
    return response.data[0].embedding


def generate_final_response(model= "ibcgpt35turbo",
                                  messages= [],
                                  temperature=0.1,
                                  max_tokens = 700,
                                  stream = True):

    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature = temperature,
                                              max_tokens = max_tokens,
                                              stream= stream,
                                             seed = 999)

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
                                                select=self.retrive_fields,top=top)  

        return results


    
    
import time


def generate_response(user_query = " ", get_sources = False,
                      search_type = "hybrid",top = 2, k =2,stream = False,max_tokens = 700):

    #print("Generating query for embedding...")
    #embedding_query = get_query_for_embedding(user_query=user_query)

    retrive_docs = retrive_similiar_docs(query = user_query)

    if search_type == "text":
        start = time.time()
        r = retrive_docs.text_search(top =top)

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
        r = retrive_docs.pure_vector_search(k=k, query_embedding = vector_of_search_query)

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
        r = retrive_docs.hybrid_search(top = top, k=k, query_embedding = vector_of_search_query)

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
if prompt := st.chat_input("Please type your query here.?"):
    # Display user message in chat message container
    st.chat_message("user",avatar = "🧑‍💻").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    
    user_question_with_prompts = prompt+dropdown_1_prompt + dropdown_2_prompt + dropdown_3_prompt

    response = generate_response(user_question_with_prompts,stream=True)   

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

