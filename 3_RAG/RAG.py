from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 단계 1: 문서 로드
loader = PyMuPDFLoader("data/[KISTEP+브리프]+‘생성형+인공지능’+시대의+10대+미래유망기술.pdf")
docs = loader.load()
print(f"문서의 페이지 수 : {len(docs)}")

# 단계 2: 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의 수 : {len(split_documents)}")

# 단계 3: 임베딩 생성
embeddings = OpenAIEmbeddings()

# 단계 4: db 생성 및 저장
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5 : 검색기 생성
retriever = vectorstore.as_retriever()

# 단계 6 : 프롬프트 생성
prompt = PromptTemplate.from_template("""
    you are an assistant for question-answering tasks.
    use the following pieces of retrieved context to answer the question.
    if you dont know the answer, just say that you dont know.
    answer in korean
                                      
    #question: {question}
    #context: {context}
    
    #answer : """)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

chain = ({"context":retriever, "question":RunnablePassthrough()}
         |prompt
         |llm
         |StrOutputParser()
         )

question = "신경망처리 기반 인공지능 전용칩에 대해 설명해주세요"
response = chain.invoke(question)
print(response)