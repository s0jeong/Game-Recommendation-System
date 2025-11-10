import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_data():
    # CF 예측 평점 행렬 (유저 x 게임)
    cf_pred = pd.read_csv("R_pred_df.csv", index_col=0)

    # CBF 메타데이터 (게임 제목, 장르 등)
    meta = pd.read_csv("cbf_meta_data_aligned.csv")

    # Game_Title, Genres만 뽑아서 우리가 쓰기 좋은 형태로 정리
    games_df = meta[["Game_Title", "Genres"]].copy()
    games_df.rename(columns={"Game_Title": "게임 이름", "Genres": "태그"}, inplace=True)

    return cf_pred, games_df

cf_pred_df, games_df = load_data()


# 1) 태그 목록 만들기

all_tags_set = set()
for tags_str in games_df["태그"]:
    if pd.isna(tags_str):
        continue
    tag_list = [t.strip() for t in str(tags_str).split(",")]
    for t in tag_list:
        if t:
            all_tags_set.add(t)

all_tags = sorted(list(all_tags_set))


# 2) 추천 함수 -> 유저 ID를 입력하면 해당 유저에게 예측된 추천 점수를 기준으로 게임 목록을 반환

def recommend_for_user(user_id: str) -> pd.DataFrame:
    
    if user_id not in cf_pred_df.index:
        return pd.DataFrame()

    # 1) 해당 유저의 예측 점수 시리즈
    user_scores = cf_pred_df.loc[user_id]

    # 2) 시리즈를 데이터프레임으로 변환
    score_df = user_scores.reset_index()
    score_df.columns = ["게임 이름", "추천 점수"]

    # 3) games_df와 합치기
    merged = score_df.merge(games_df, on="게임 이름", how="left")

    # 4) 추천 점수 높은 순으로 정렬
    merged = merged.sort_values("추천 점수", ascending=False)

    return merged


# 3) Streamlit UI

st.title("🎮 유저 평가 기반 게임 추천")
st.write("""
유저 ID를 선택하면 해당 유저에게 예측된 평점을 기준으로
게임 추천 리스트를 보여줍니다.
검색과 태그 필터도 함께 사용할 수 있어요.
""")

# --- 3-1) 사이드바: 유저 ID, 검색, 태그 선택 ---
with st.sidebar:
    st.header("필터 설정")

    # (1) 유저 ID 선택
    user_ids = cf_pred_df.index.tolist()
    user_id_selected = st.selectbox(
        "유저 ID 선택",
        options=["(선택 안함)"] + user_ids,
        index=0
    )

    # (2) 게임 이름 검색창
    search_query = st.text_input(
        "게임 이름 검색",
        value="",
        placeholder="게임 이름 일부를 입력해보세요."
    )

    # (3) 태그 멀티 선택
    selected_tags = st.multiselect(
        "태그 선택 (다수 선택 가능)",
        options=all_tags,
        default=[]
    )

# --- 3-2) 유저 ID 여부에 따라 기본 데이터 결정 ---

if user_id_selected != "(선택 안함)":
    base_df = recommend_for_user(user_id_selected)
    if base_df.empty:
        st.error(f"유저 ID `{user_id_selected}` 에 해당하는 추천 데이터를 찾을 수 없습니다.")
    else:
        st.info(f"유저 ID `{user_id_selected}` 기준 추천 순서입니다.")
else:
    # 유저 ID 선택 안 한 경우: 그냥 게임 이름 가나다순으로 보여주기
    base_df = games_df.copy()
    base_df["추천 점수"] = np.nan
    base_df = base_df.sort_values("게임 이름")
    st.info("유저 ID를 선택하지 않아, 게임 이름 가나다순으로 보여줍니다.")


# --- 3-3) 검색 + 태그 필터 적용 ---

filtered_df = base_df.copy()

# (1) 게임 이름 검색어 필터
if search_query.strip():
    filtered_df = filtered_df[
        filtered_df["게임 이름"].str.contains(search_query, case=False, na=False)
    ]

# (2) 태그 필터
if selected_tags:
    def has_selected_tag(tag_str):
        if pd.isna(tag_str):
            return False
        tag_list = [t.strip() for t in str(tag_str).split(",")]
        return any(t in tag_list for t in selected_tags)

    filtered_df = filtered_df[filtered_df["태그"].apply(has_selected_tag)]


# --- 3-4) 결과 보여주기 ---

st.write(f"현재 조건에 맞는 게임 수: **{len(filtered_df)}개**")

if filtered_df.empty:
    st.warning("조건에 맞는 게임이 없습니다. 유저 ID, 검색어, 태그를 바꿔보세요.")
else:
    # 보기 좋은 순서로 컬럼 정리
    cols_to_show = ["게임 이름", "태그"]
    if "추천 점수" in filtered_df.columns:
        cols_to_show.append("추천 점수")

    st.dataframe(
        filtered_df[cols_to_show],
        use_container_width=True,
        height=500
    )
