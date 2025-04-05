import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
#from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.memory import ConversationEntityMemory


# 1. 문서 불러오기 (텍스트 파일 예시)
loader = TextLoader("sample.txt", encoding="utf-8")  # <- 너가 가진 파일
docs = loader.load()

# 2. 문서 분할
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# 3. 임베딩
embeddings = OpenAIEmbeddings()

# 4. FAISS 인덱스 만들기
db = FAISS.from_documents(documents, embeddings)

# 5. 저장하기
db.save_local("your_faiss_index")  # 저장할 폴더명


# 환경변수 로드 (API KEY 등)
load_dotenv()

# --- 기본 설정 ---
st.title("❤️나만의 첫GPT 요약기❤️")

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 사이드바 설정 ---
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    if clear_btn:
        st.session_state.messages = []
    selected_prompt = st.selectbox(
        "프로머트를 선택해 주세요.",
        ("기본메시지", "SNS 계정 게시물", "요약 프로머트"),
        index=0
    )


# --- 이전 대화 출력 ---
def print_messages():
    for chat in st.session_state.messages:
        st.chat_message(chat["role"]).write(chat["content"])

# --- 대화 저장 ---
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# --- 프롬프트 기반 체인 생성 ---
def create_chain(prompt_type):
    if prompt_type == "SNS 계정 게시물":
        try:
            prompt = load_prompt("prompts/sns.yaml", encoding="cp949")
        except:
            prompt = load_prompt("prompts/sns.yaml", encoding="utf-8")
    elif prompt_type == "요약 프로머트":
        try:
            prompt = load_prompt("prompts/summary.yaml", encoding="cp949")
        except:
            prompt = load_prompt("prompts/summary.yaml", encoding="utf-8")
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a kind AI assistant. Please answer the following questions concisely."),
            ("user", "#Question:\n{question}")
        ])

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser


llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
conversation = ConversationChain(
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm),
)

# --- 메인 흐름 ---
print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.chat_message("user").write(user_input)
    add_message("user", user_input)

    if selected_prompt == "기본메시지":
        # 기본 모드는 memory + retriever (qa_chain)
        result = conversation.invoke({"input": user_input})
        ai_answer = result["response"]
        st.chat_message("assistant").write(ai_answer)
    else:
        # 다른 모드는 직접 체인 만들어 사용
        chain = create_chain(selected_prompt)
        response = chain.stream({"question": user_input})
        ai_answer = ""
        with st.chat_message("assistant"):
            container = st.empty()
            for chunk in response:
                ai_answer += chunk
                container.markdown(ai_answer)

    add_message("assistant", ai_answer)
