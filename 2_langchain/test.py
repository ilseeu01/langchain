import streamlit as st
from langchain_core.messages.chat import ChatMessage

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
import pandas as pd
import base64
import requests

opencage_key = os.getenv("OPENCAGE_API_KEY")


st.title("☔ 도시와 날씨를 지도에서 보기 ☀️")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대회기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 사이드 바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")
    if clear_btn:
        # st.write("버튼이 눌렸습니다.")
        st.session_state["messages"] = []

# 이미지 파일을 base64로 읽기
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 배경 CSS 세팅
def set_bg_from_local(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_bg_from_local('clouds.jpg')

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 세로운 메시지를 추가
def add_messages(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인 생성
def create_chain():
    # prompt | llm | output+parser
    # 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절한 AI 어시스턴트입니다"),
            ("user", "#Question:\n{question}"),
        ]
    )
    # 모델
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    # 출력파서
    output_parser = StrOutputParser()

    # 체인생성
    chain = prompt | llm | output_parser
    return chain


st.markdown("""
<style>
h1 {
  color: white;
  text-shadow: 2px 2px 4px #000000;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stChatInput"] {
    background-color: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(5px);
    border-radius: 999px; /* 완전히 둥글게 */
    padding: 10px;
    
}
[data-testid="stChatInput"] textarea {
    border-radius: 999px;
    padding: 10px;
    background-color: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(5px);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stChatInput"] textarea::placeholder {
    color: #666666; /* 글자 색 */
    font-size: 16px; /* 글자 크기 */
    font-weight: 500; /* 글자 굵기 */
    opacity: 1; /* 투명도 줄이기 (1=선명) */
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
section.main > div:has(.stChatInput) {
    background-color: rgba(0,0,0,0) !important; /* 완전 투명 */
    border-radius: 999px;
}
</style>
""", unsafe_allow_html=True)

print_messages()


def get_coordinates_from_city(city_name, api_key):
    url = f"https://api.opencagedata.com/geocode/v1/json"
    params = {
        'q': city_name,
        'key': api_key,
        'language': 'ko',
        'limit': 1
    }

    res = requests.get(url, params=params).json()
    if res["results"]:
        lat = res["results"][0]["geometry"]["lat"]
        lon = res["results"][0]["geometry"]["lng"]
        return lat, lon
    else:
        return None, None

# 사용자의 입력
user_input = st.chat_input("궁금한 것을 물어보세요! 날씨도 제공해드릴 수 있어요요")

# 만약 사용자 입력이 들어오면
if user_input:
    st.chat_message("user").write(user_input)

    # 기본 질문
    question_for_llm = user_input

    
    # 날씨 질문이면 API 정보 추가
    # 날씨 관련 키워드가 들어있다면
    if "날씨" in user_input or "기온" in user_input or "비" in user_input:
    
        # 도시 추출
        extract_chain = create_chain()
        city_name = extract_chain.invoke({"question": f"{user_input}에서 도시 이름만 한 단어로 추출해줘. 없으면 '없음'이라고 말해줘."}).strip()

        if city_name != "없음":
            lat, lon = get_coordinates_from_city(city_name, opencage_key)

            # 날씨 API 호출
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,weathercode",
                "timezone": "Asia/Seoul"
            }
 
            day_index = 1  # 기본: 내일
            if "오늘" in user_input:
                day_index = 0
            elif "모레" in user_input:
                day_index = 2
            res = requests.get(url, params=params).json()
            # 날씨 정보 추출
            temp_max = res["daily"]["temperature_2m_max"][day_index]
            temp_min = res["daily"]["temperature_2m_min"][day_index]
            rain = res["daily"]["precipitation_sum"][day_index]
            snowfall = res["daily"]["snowfall_sum"][day_index]
            weathercode = res["daily"]["weathercode"][day_index]
            target_date = res["daily"]["time"][day_index]
            print("눈 확률 : ", snowfall, weathercode)

            weather_info = f"{city_name}의 {target_date} 날씨는 최저 {temp_min}도, 최고 {temp_max}도이며, 강수량은 {rain}mm입니다."
            

            # 눈 오는지 판단
            if snowfall > 0 or weathercode in [71, 73, 75, 77, 85, 86]:
                st.snow()
                weather_info += " 눈이 내릴 가능성이 높습니다."


            if lat and lon:
                st.success(f"{user_input}를 지도에 표시합니다!")
                df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
                st.map(df, zoom=5)  # 중심은 해당 좌표, 줌 레벨은 10
            
            
            # 프롬프트에 날씨 포함
            question_for_llm = f"""
            사용자 질문: {user_input}
            날씨 정보: {weather_info}
            이 날씨 정보를 바탕으로 자연스럽고 친절하게 대답해줘.
            """

            



    # 체인 실행은 딱 한 번만!
    chain = create_chain()
    response = chain.stream({"question": question_for_llm})

    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    add_messages("user", user_input)
    add_messages("assistant", ai_answer)
    