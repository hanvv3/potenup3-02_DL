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

