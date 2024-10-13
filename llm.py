from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain import hub 
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def get_retriever():
    embedding = OpenAIEmbeddings(model = "text-embedding-3-large")
    index_name = "table-markdown-index"
    database = PineconeVectorStore.from_existing_index(
        embedding=embedding,
        index_name=index_name
    )
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


def get_llm(model="gpt-4o"):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        사전: {dictionary}

        질문: {{question}}                                      

        """)

    dictionary_chain = prompt | llm | StrOutputParser()   
    return dictionary_chain


def get_qa_chain():
    llm = get_llm()
    retriever = get_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type_kwargs={"prompt": prompt},
        retriever=retriever
    )
    return qa_chain


def get_ai_response(user_message):

    dictionary_chain = get_dictionary_chain()
    qa_chain = get_qa_chain()
    tax_chain = {"query" : dictionary_chain} | qa_chain

    ai_message = tax_chain.invoke({"question": user_message})
    return ai_message["result"]
