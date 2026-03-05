import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import streamlit as st

from PIL import Image

st.title('Hello Streamlit!')
st.write('This is my first Streamlit page.')

with st.form("greet_form"):
    col1, col2 = st.columns([4, 1])  # 비율 조정

    with col1:
        name = st.text_input('Put your name.', label_visibility="collapsed")

    with col2:
        submitted = st.form_submit_button('Greet')

if submitted:
    st.success(f'Hello {name}, nice to meet you.')

st.markdown("----------------------------------")

st.title("스트림릿 제목")
st.header("헤더")
st.subheader("서브헤더")
st.text("일반 텍스트")
st.markdown("**마크다운 지원** :sparkles:")
st.code("print('Hello World')", language="python")

col1, col2 = st.columns(2)
col1.write("왼쪽 컬럼")
col2.write("오른쪽 컬럼")

with st.expander("펼치기/접기"):
    st.write("숨겨진 내용")

age = st.number_input("나이 입력", min_value=0, max_value=120, value=25)
score = st.slider("점수", 0, 100, 50)

agree = st.checkbox("동의합니다")
option = st.radio("좋아하는 색상", ["빨강", "파랑", "초록"])
select = st.selectbox("과목 선택", ["수학", "과학", "영어"])
multi = st.multiselect("취미 선택", ["독서", "운동", "게임"])

uploaded_file = st.file_uploader("파일 업로드", type=["jpg","png","csv","webp"])

if uploaded_file:
    st.image(uploaded_file)

img = Image.open("./data/Celeb/train/뉴진스 해린/Image_64.jpeg")
st.image(img, caption="고양이", width="content")

#st.audio("music.mp3")
#st.video("video.mp4")

df = pd.DataFrame({"이름":["철수","영희"], "점수":[90,80]})
st.table(df)       # 정적 테이블
st.dataframe(df)   # 인터랙티브 테이블

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a","b","c"])
st.line_chart(chart_data)
st.bar_chart(chart_data)
st.area_chart(chart_data)

progress = st.progress(0)
for i in range(100):
    time.sleep(0.05)
    progress.progress(i+1)

st.success("성공 메시지")
st.error("에러 메시지")
st.warning("경고 메시지")
st.info("정보 메시지")

st.markdown("----------------------------------")
### 시각화 ###
st.header("시각화")

df = pd.DataFrame({
    "이름": ["철수", "영희", "민수"],
    "점수": [90, 85, 70]
})

st.write("학생 점수 데이터")
st.dataframe(df)  # 스크롤, 정렬 가능한 테이블
st.table(df)      # 정적인 테이블

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["a", "b", "c"]
)

st.line_chart(chart_data)   # 선 그래프
st.area_chart(chart_data)   # 영역 그래프
st.bar_chart(chart_data)    # 막대 그래프

## matplotlib.pyplot
fig, ax = plt.subplots()
ax.hist(df["점수"], bins=5)
st.pyplot(fig)

## plotly
fig = px.scatter(chart_data, x="a", y="b", color="c")
st.plotly_chart(fig)

## altair
c = alt.Chart(chart_data).mark_circle().encode(
    x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"]
)
st.altair_chart(c, use_container_width=True)

# 가짜 데이터 (학습 과정 시뮬레이션)
epochs = list(range(1, 11))
loss = [1/x for x in epochs]
accuracy = [x*10 for x in epochs]

fig, ax = plt.subplots()
ax.plot(epochs, loss, label="Loss")
ax.plot(epochs, accuracy, label="Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)

st.markdown("----------------------------------")

### 0) 기본 설정 (와이드 모드)
#### layout="wide": 더 넓은 화면에서 대시보드형 UI 구성에 유리
st.set_page_config(page_title="레이아웃 데모", layout="wide")

### 1) 사이드바(sidebar)와 메인 영역
#### 패턴: 입력(필터)은 sidebar, 결과는 메인에 표시 → 사용성이 좋음
with st.sidebar:
    st.header("필터")
    date = st.date_input("날짜")
    cls = st.selectbox("클래스", ["전체","A","B"])

st.title("대시보드")
st.write("사이드바에서 필터를 조정하세요.")

### 2) 컬럼(columns)로 가로 배치
#### 꿀팁: 리스트로 비율 지정 → 반응형 + 균형 잡힌 배치
col1, col2, col3 = st.columns([2, 3, 2])  # 비율로 너비 제어
with col1:
    st.subheader("요약 지표")
    st.metric("Accuracy", "93.2%", "+0.7%")
with col2:
    st.subheader("라인 차트")
    st.line_chart({"acc":[0.8,0.85,0.9,0.932]})
with col3:
    st.subheader("세부 옵션")
    st.checkbox("스무딩")

### 3) 탭(tabs)으로 화면 전환
#### 패턴: “개요/지표/원본데이터(로그)” 3단 탭은 수업에서 아주 인기
tab1, tab2, tab3 = st.tabs(["개요", "지표", "로그"])
with tab1:
    st.write("한 눈에 보는 개요")
with tab2:
    st.write("정밀 지표 표/차트")
with tab3:
    st.code("학습 로그 미리보기...")

### 4) 확장(expander)로 추가 정보 접기/펼치기
#### 본문을 깔끔하게 유지하면서 부가 설명 제공
with st.expander("전처리 설명 보기"):
    st.markdown("- 이진화 → 자르기 → 패딩 → 28x28 리사이즈")

### 5) 컨테이너(container)와 플레이스홀더(placeholder)
#### 패턴: 로딩/스트리밍 진행 상태 표시, 실시간 업데이트 UI
placeholder = st.empty()        # 화면의 빈 자리 확보
with st.container():
    st.write("섹션 시작")
    st.write("여러 컴포넌트를 묶어 관리")
# 동적으로 업데이트
placeholder.metric("현재 단계", "로딩 중...")
# ... 처리 후
placeholder.metric("현재 단계", "완료")


### 6) 폼(form)으로 입력 묶기 (Submit까지 한 번에)
#### 장점: 여러 입력을 한 번에 제출 → 예측/학습 파라미터 설정에 딱
with st.form("hyperparams"):
    lr = st.number_input("Learning Rate", 0.0001, 1.0, 0.001, format="%.4f")
    epochs = st.slider("Epochs", 1, 200, 30)
    submitted = st.form_submit_button("학습 시작")
if submitted:
    st.success(f"LR={lr}, Epochs={epochs}로 학습 시작!")

### 7) 세션 상태(session_state)로 상호작용 기억
#### 활용: 탭 전환/재실행에도 선택값 유지
'''
if "filters" not in st.session_state:
    st.session_state.filters = {"cls":"전체"}

st.session_state.filters["cls"] = st.sidebar.selectbox(
    "클래스", ["전체","A","B"], index=["전체","A","B"].index(st.session_state.filters["cls"])
)

st.write("선택된 클래스:", st.session_state.filters["cls"])
'''
# 스트림릿(streamlit)의 st.session_state는 웹 앱이 매번 리렌더링(스크립트 재실행) 될 때도
# 사용자의 상태(state)를 기억하는 저장소입니다.
# 왜 필요한가?
# - **기본 동작**: Streamlit은 사용자가 버튼 클릭, 슬라이더 조정 등 이벤트를 발생시키면 전체 스크립트를 위에서부터 다시 실행합니다.
# - 이때 일반 변수는 초기화되므로 값이 유지되지 않습니다.
# - → `st.session_state`를 사용하면 **사용자 인터랙션, 파라미터, 상태**를 보존할 수 있습니다.


########################### 템플릿 ################################
#import streamlit as st
st.set_page_config(page_title="모델 대시보드", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.header("필터")
    dataset = st.selectbox("데이터셋", ["Val","Test"])
    smooth = st.slider("스무딩", 1, 25, 5)
    show_points = st.checkbox("포인트 표시", False)

# --- Header ---
st.title("모델 성능 대시보드")
st.caption("실험 비교 · 지표 요약 · 예측 분포")

# --- KPI Row ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Best Val Acc", "93.2%")
k2.metric("Min Val Loss", "0.183")
k3.metric("Latency(ms)", "12.4")
k4.metric("Params(M)", "21.8")

# --- Tabs ---
t1, t2, t3 = st.tabs(["학습 곡선", "지표표/혼동행렬", "예측 샘플"])

with t1:
    c1, c2 = st.columns([3,2])
    with c1:
        st.subheader("Loss Curve")
        st.line_chart({"train":[0.9,0.6,0.4,0.25], "val":[1.0,0.7,0.5,0.3]})
    with c2:
        st.subheader("Accuracy Curve")
        st.line_chart({"train":[0.5,0.7,0.85,0.92], "val":[0.45,0.65,0.8,0.9]})

with t2:
    st.subheader("지표 테이블")
    st.table({"precision":[0.91,0.88], "recall":[0.90,0.86], "f1":[0.905,0.87]})
    with st.expander("혼동행렬 보기"):
        st.dataframe({"Pred 0":[88,5], "Pred 1":[7,100]})

with t3:
    left, right = st.columns([2,3])
    with left:
        st.subheader("입력 샘플")
        st.image("https://placehold.co/300x300", caption="업로드/샘플")
    with right:
        st.subheader("Top-k 확률")
        st.bar_chart({"A":[0.7], "B":[0.2], "C":[0.1]})


### 9) 레이아웃 설계 베스트 프랙티스

# - **정보 구조 먼저**: “필터 → KPI(요약) → 상세 탭(곡선/표/원본)” 순으로
# - **한 화면 = 한 목적**: 탭으로 역할 분리(개요/지표/원본/로그)
# - **시각적 균형**: `columns([3,2])`처럼 비율로 그리드 잡기
# - **설명은 expander**에 넣어 본문 간결화
# - **성능 고려**: 무거운 계산은 `st.cache_data/resource`로 캐시
# - **재활용**: 템플릿(위 8번)에서 컴포넌트만 갈아끼기