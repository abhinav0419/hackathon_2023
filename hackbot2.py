# %%
# Integrate Modern Data Architectures with Generative AI and interact using prompts for querying SQL databases & APIs

import json
import boto3
import streamlit as st
import os
import sqlalchemy

from sql_tools import CustomSQLDatabaseToolkit
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from langchain.agents import AgentType, create_sql_agent

#import streamlit packages

# assign your llm and db
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.docstore.document import Document
from langchain.llms import SagemakerEndpoint
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain
from langchain.llms.bedrock import Bedrock
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import FewShotPromptTemplate
#from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate, FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chat_models import ChatAnthropic

from typing import Dict

# %%
#Define Glue for 

# %%

glue_crawler_name = '2023_hackathon_ruth_group_finance' #params['CFNCrawlerName'] 
glue_database_name = '2023_hackathon_sfdc' #params['CFNDatabaseName']
glue_databucket_name = 'finance_twitch_finance' #params['DataBucketName']
region = 'us-east-1' #outputs['Region']

# %% [markdown]
# **Important**: The code below establishes a database connection for data sources and Large Language Models. Please note that the solution will only work if the database connection for your sources is defined in the cell below. Please refer to the Pre-requisites section. If your use case requires data from Aurora MySQL alone, then please comment out other data sources. Furthermore, please update the cluster details and variables for Aurora MySQL accordingly.

# %%



#S3
# connect to s3 using athena
## athena variables
connathena=f"athena.{region}.amazonaws.com" 
portathena='443' #Update, if port is different
schemaathena='2023_hackathon_sfdc' #from cfn params
s3stagingathena=f's3://2023-ruth-hackathon-group-gen-ai/athenaresults/'#from cfn params
wkgrpathena='primary'#Update, if workgroup is different
tablesathena=['finance_twitch_finance'] #[<tabe name>]
##  Create the athena connection string
connection_string = f"awsathena+rest://@{connathena}:{portathena}/{schemaathena}?s3_staging_dir={s3stagingathena}/&work_group={wkgrpathena}"
# print(connection_string)
##  Create the athena  SQLAlchemy engine
engine_athena = create_engine(connection_string, echo=False)
dbathena = SQLDatabase(engine_athena)
db = dbathena
#Glue Data Catalog
##Provide list of all the databases where the table metadata resides after the glue successfully crawls the table
# gdc = ['redshift-sagemaker-sample-data-dev', 'snowflake','rds-aurora-mysql-employees','sagemaker_featurestore'] # mentioned a few examples here
gdc = [schemaathena] 
print(gdc)

# %% [markdown]
# ### Step 2 - Generate Dynamic Prompt Templates
# Build a consolidated view of Glue Data Catalog by combining metadata stored for all the databases in pipe delimited format.

# %%
#Generate Dynamic prompts to populate the Glue Data Catalog
#harvest aws crawler metadata

def parse_catalog():
    #Connect to Glue catalog
    #get metadata of redshift serverless tables
    columns_str=''
    #define glue cient
    glue_client = boto3.client('glue')
    
    for db in gdc:
        response = glue_client.get_tables(DatabaseName =db)
        #return response
        for tables in response['TableList']:
            for columns in tables['StorageDescriptor']['Columns']:
                    dbname,tblname,colname=tables['DatabaseName'],tables['Name'],columns['Name']
                    columns_str=columns_str+f'{dbname}|{tblname}|{colname}\n'                     
    #API
    ## Append the metadata of the API to the unified glue data catalog
    columns_str=columns_str+'\n '
    return columns_str

glue_catalog = parse_catalog()

# print(glue_catalog)
# display a few lines from the catalog
# print('\n'.join(glue_catalog.splitlines()[-10:]) )


# %%


# %%
### Step 3 - Define Functions to 1/ determine the best data channel to answer the user query, 2/ Generate response to  user query

# %%
#In this code sample, we use the Anthropic Model to generate inferences. You can utilize SageMaker JumpStart models  to achieve the same. 
#Guidance on how to use the JumpStart Models is available in the notebook - mda_with_llm_langchain_smjumpstart_flant5xl

# %%

#INITIALIZE BEDROCK CLIENT
bedrock_runtime = boto3.client(service_name='bedrock-runtime')

#LLM 
#llm = Bedrock(model_id="amazon.titan-text-express-v1")
#llm = Bedrock(model_id = "anthropic.claude-v2")
llm = Bedrock(model_id="ai21.j2-ultra-v1", model_kwargs={"maxTokens": 1024,"temperature": 0.2})


#def identify_table(query):
#    prompt_template = """
#    From the table below, find the database (in column database) which will contain the data (in corresponding column_names) to answer the question {query} \n
#    """+glue_catalog +""" Give your answer as database == \n Also,give your answer as database.table =="""
#
#    PROMPT = PromptTemplate(template=prompt_template, input_variables=["query"])
#    # define llm chain
#    llm_chain = LLMChain(prompt=PROMPT, llm=llm)
#    generated_texts = llm_chain.run(query)
#    return generated_texts
#
#    Given an input question, first create a syntactically correct athena query to run, then look at the results of the query and return the answer.
#    Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
#    If someone asks for the revenue, they really mean the finance_twitch_finance table.
#    If someone asks for metrics be sure to return the value in a human readable response
#    Use only the finance_twitch_finance table
_mysql_template = """
You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURDATE() function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here
"""
#def run_query(query):
##    query_tables = identify_table(query)
#    _DEFAULT_TEMPLATE = """
#    Create a syntactically correct athena query to run based on the question
#    
#    First retrieve the table name, table columns
#    Then check the query to ensure all columns are part of their respective tables.
#    Then check the query to ensure it follows correct sql syntax.
#    Use only the finance_twitch_finance
#    {table_info}
#    
#    Question: {input}
#   
#    Context:
#    Use local currency column (ie. lc_reported_revenue) if question requests for currency
#    Use usd_reported_revenue if question requets usd as currency
#    the column report_year refers to the year the revenue was recorded
#    the column report_month refers to the month the revenue was recorded    
#    
#    Respond to the answer in a human readable sentence, if using numbers format to the nearest whole number
#           """
#    PROMPT = PromptTemplate(template=_mysql_template, input_variables=["input","table_info"])
#    db = dbathena    
#    db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose = True,
#                                         use_query_checker=True, # self-correcting small mistakes NOT WORKING
#                                         top_k=3 # limit the number of rows returned
#                                        )
#    response=db_chain.run(query)
#    return response



# %%
# In This Code we are testing SQL Database Agent


examples = [
        {
            "input": "Which account has the highest revenue in 2023?",
            "sql_cmd": """SELECT salesforce_account_name, sum(usd_reported_revenue) as revenue 
                          FROM finance_twitch_finance
                          Where report_year = 2023
                          Group By 1
                          Order by 2 desc
                          Limit 1
                    ;""",
            "result": "[Apple - Global, (100000000,)]",
            "answer": "Apple - Global has an 2023 annual revenue of $100,000,000",
        },
        {
            "input": "Which account has the highest revenue in the entertainment sales_vertical in 2023?",
            "sql_cmd":"""SELECT sales_vertical, salesforce_account_name, sum(usd_reported_revenue) as revenue 
                          FROM finance_twitch_finance
                          Where report_year = 2023 and sales_vertical = 'Entertainment'
                          Group By 1,2
                          Order by 2 desc
                          Limit 1
                    ;""",
            "result": "[Entertainment, Apple - Global, (100000000,)]",
            "answer": "Apple - Global has the highest 2023 annual revenue in the Entertainment sales_vertical with $100,000,000",
        },
        {
            "input": "Which product did Apple - Global spend the most revenue on in 2023",
            "sql_cmd":"""SELECT salesforce_account_name, property_name, sum(usd_reported_revenue) as revenue 
                          FROM finance_twitch_finance
                          Where report_year = 2023 and salesforce_account_name = 'Apple - Global'
                          Group By 1,2
                          Order by 3 desc
                          Limit 1
                    ;""",
            "result": "[Audio Ads, (30000,)]",
            "answer": "Apple - Global spent the most revenue on 'Audio Ads' with a 2023 annual spend of $30,000",
        },
]

_mysql_prompt_ = """
You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURDATE() function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here
"""

#example_prompt = PromptTemplate(
#    input_variables=["input", "sql_cmd", "result", "answer",],
#    template="\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {result}\nAnswer: {answer}",
#)

#example_prompt = ChatPromptTemplate.from_messages(
#    [
#        ("human", "{input}"),
#        ("ai", "\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {result}\nAnswer: {answer}"),
#    ]
#)

example_prompt = PromptTemplate(
    input_variables=["input", "sql_cmd", "result", "answer",],
    template="\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {result}\nAnswer: {answer}",
)

embeddings = HuggingFaceEmbeddings()

to_vectorize = [" ".join(example.values()) for example in examples]

vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=1,
)


#few_shot_prompt = FewShotChatMessagePromptTemplate(
#    example_selector=example_selector,
#    example_prompt=example_prompt,
#    #prefix=_mysql_prompt_,
#    #suffix="Question: {input}",
#    #suffix=prompt_suffix, 
#    input_variables=["input"], #These variables are used in the prefix and suffix
#)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=_mysql_prompt_,
    suffix="Question: {input}",
    #suffix=prompt_suffix, 
    input_variables=["input"], #These variables are used in the prefix and suffix
)


# Enable Conversation Buffer Memory
msgs = StreamlitChatMessageHistory(key="special_app_key")

memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")
    
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)    

#sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm , use_query_checker=True)
#sql_toolkit = CustomSQLDatabaseToolkit(db=db, llm=llm , use_query_checker=True)
#sql_toolkit.get_tools()

#sqldb_agent = create_sql_agent(
#    llm=llm,
#    toolkit = sql_toolkit,
#    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#    verbose=True,
#    handle_parsing_errors=True,
#    example_selector=example_selector,
#    memory=memory
#)



#final_prompt = ChatPromptTemplate.from_messages(
#    [ ("system", _mysql_prompt_),
#     few_shot_prompt,
#      ("human", "{input}"),
#    ]
#)
#
#
#
#final_prompt = ChatPromptTemplate.from_messages(
#    [ ("system",
#       """
#        Create a syntactically correct athena query to run based on the question
#    
#        First retrieve the table name, table columns
#        Then check the query to ensure all columns are part of their respective tables.
#        Then check the query to ensure it follows correct sql syntax.
#        Use only the finance_twitch_finance
#
#        Context:
#        Use local currency column (ie. lc_reported_revenue) if question requests for currency
#        Use usd_reported_revenue if question requets usd as currency
#        the column report_year refers to the year the revenue was recorded
#        the column report_month refers to the month the revenue was recorded    
#
#        Respond to the answer in a human readable sentence, if using numbers format to the nearest whole number
#        
#        """),
#     ("user", "{question}\n ai: ")
#    ]
#)
#    
#
## %%
## Run Queries
#query_01 = """Which product reported the most revenue and how much?""" 
#query_02 = """Which Account reported the most total revenue in 2023?""" 
#query_03 = """Can you compare top 2 unique account by total revenue?"""
#
# response = run_query(query_01)
# print(response)


# sqldb_agent.run(final_prompt.format( question=query_01 ))
# Enable Chatbot based on Prompt    
if prompt := st.chat_input():    
    st.chat_message("user").write(prompt)
    with st.chat_message("system"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = local_chain.run(prompt, callbacks=[st_callback])
        st.write(response)
