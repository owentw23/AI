import openai
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults


#Loading Sources
llm = Ollama(model="llama2")
urls = ["https://books.google.com.sg/books?hl=en&lr=&id=eEYrDwAAQBAJ&oi=fnd&pg=PP1&dq=urban+density+and+sustainability&ots=Vebg2OIzPf&sig=e9h3t9qTqyGhSUvoR9uq9zcYJ_E&redir_esc=y#v=onepage&q=urban%20density%20and%20sustainability&f=false", "https://www.sciencedirect.com/science/article/pii/S1470160X20309687", "https://bit.ly/airqualcitydens", "https://www.nature.com/articles/s41598-020-74524-9", "https://www.health.act.gov.au/about-our-health-system/population-health/environmental-monitoring/air-quality/measuring-air#:~:text=The%20AQI%20is%20calculated%20from,Ambient%20Air%20(NEPM)%20standard.", "https://www.sciencedirect.com/topics/social-sciences/building-density", "https://ourworldindata.org/urbanization#:~:text=may%20lack%20several.-,How%20is%20urban%20density%20defined%3F,area%20for%20that%20given%20population.", "https://databank.worldbank.org/reports.aspx?source=2&series=SP.URB.TOTL.IN.ZS&country="]
loader = WebBaseLoader(urls)
embeddings = OllamaEmbeddings()
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

#Agent
retriever_tool = create_retriever_tool(
    retriever,
    "climatechange_search",
    "Search for information about urban infrastructure and climate condition and change. For any questions about climate or urban structuring plans, you must use this tool!",)
tools = [retriever_tool]

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# Get the prompt to use - you can modify this!

#Chat history Chain
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

chat_history = [HumanMessage(content="Is there a relationship between climate conditions and urban infrastructure density?"), AIMessage(content="Yes!")]
retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

#Prompt Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context} Base your answer completely on the provided documents and datasets, say you cannot answer the question if you cannot come up with a concrete answer. Be more human. DO NOT ANSWER ANY QUESTION OUTSIDE THE TOPIC OF SUSTAINABILITY, URBAN INFRASTRUCTURE BUILDING, AND CLIMATE CHANGE. Make use of mathematical methods if needed. DO NOT ACCESS THE INTERNET FOR INFORMATION BESIDES THE PROVIDED INFORMATION FROM THE DOCUMENTS. Only give a summary if prompted to do so, or you are asked to make a long explanation. When giving an answer give 1 fixed answer please, do not give a list of possible answers but rather give 1 concrete and fixed answer. "),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

chain = retrieval_chain

#Chat

def process_chat(chains, question, chathistory):
    response = chains.invoke({"chat_history": chathistory, "input": question})
    return response["answer"]

chat_history = []

def qna(user_input):
    if user_input.lower() == 'exit':
        return 0
    response = process_chat(chain, user_input, chat_history)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    return response

if __name__ == "__main__":
    #Loop
    while True:
       user_input = input("You: ")
       answer = qna(user_input)
       print("Assistant: ", answer)