# steam_test_model.py (최종 완성 버전 - Steam CF 통합 및 오류 해결)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re
import warnings
import os
import ast
from sklearn.decomposition import TruncatedSVD # Steam CF 학습용

# --------------------------------------------------------------------------------
# 주의: 이 코드를 실행하려면 cbf_module.py, cf_module.py, normalize_module.py, cluster_module.py 
# 네 가지 모듈 파일이 동일 디렉토리에 있어야 합니다.
# --------------------------------------------------------------------------------
try:
    from cbf_module import build_cbf_model, clean_title_for_match
    from cf_module import build_cf_model
    from normalize_module import normalize_matrix, calculate_rmse
    from cluster_module import tune_kmeans_k, apply_clustering_and_calculate_affinity
except ImportError as e:
    print(f"\n [치명적 오류] 모듈 임포트 실패: {e}")
    print("네 가지 모듈 파일(`.py`)이 현재 디렉토리에 모두 존재하는지 확인하세요.")
    exit()

# --- 0. 설정 ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# **로컬 경로 설정**
BASE_DIR = "./" 
RESULTS_DIR = "./results/"

# 하이퍼파라미터 설정
FINAL_N_CLUSTERS = 30                 # 최종 사용할 K 값 
TEST_SET_RATIO = 0.2                  # Steam 매트릭스의 테스트 데이터 비율 (20%)

# 결과 저장 폴더 생성 (로컬 환경 지원)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f" [정보] 결과 폴더 '{RESULTS_DIR}' 생성 완료.")

# OpenCritic 데이터 (Train)
PATH_CF_DATA = BASE_DIR + "density_processed.csv"
PATH_META_DATA_OC = BASE_DIR + "metacritic_light_content_metadata.csv"

# Steam 데이터 (Test)
PATH_META_DATA_STEAM = BASE_DIR + "game_metadata.csv"        
PATH_REVIEW_DATA_STEAM = BASE_DIR + "review_data.csv"         
PATH_USER_MATRIX_STEAM = BASE_DIR + "user_game_review_matrix.csv" 


# --- 1. Steam 데이터 전처리 및 통합 함수 ---

def preprocess_steam_data(path_meta, path_review, path_matrix):
    
    print("--- [Steam Preprocess] 1. 데이터 로드 및 오류 처리 ---")
    
    try:
        df_meta = pd.read_csv(path_meta, encoding='utf-8-sig')
        df_reviews = pd.read_csv(path_review, encoding='utf-8-sig')
        df_matrix = pd.read_csv(path_matrix, index_col=0, encoding='utf-8-sig')
    except (FileNotFoundError, Exception) as e:
        print(f" [치명적 오류] 파일 로드 실패: {e}")
        return None, None
    
    df_reviews['review_text'].fillna('', inplace=True) 

    # 2. 콘텐츠 통합
    df_review_text_agg = df_reviews.groupby('AppID')['review_text'].agg(lambda x: ' '.join(x)).reset_index()
    df_review_text_agg.rename(columns={'review_text': 'Aggregated_Review_Text'}, inplace=True)
    df_meta_steam = pd.merge(df_meta, df_review_text_agg, on='AppID', how='left')
    df_meta_steam.fillna({'Aggregated_Review_Text': ''}, inplace=True) 
    
    # 3. CBF 포맷팅 및 이름 정규화
    df_meta_steam['Title'] = df_meta_steam['name'].apply(clean_title_for_match)
    df_meta_steam['Source'] = 'Steam'
    
    def clean_genre_list_str(genre_str):
        if pd.isna(genre_str) or genre_str == '[]': return ''
        try:
            genres = ast.literal_eval(genre_str) 
            return ' '.join(genres)
        except (ValueError, TypeError, SyntaxError):
            return ''
            
    df_meta_steam['Genres'] = df_meta_steam['genres'].apply(clean_genre_list_str)
    df_meta_steam['Description'] = df_meta_steam['Aggregated_Review_Text'] 
    df_meta_steam = df_meta_steam[['Title', 'Genres', 'Description', 'Source', 'AppID']].copy()

    # 4. 평점 행렬 정규화 (-1~1 -> 0~100)
    df_matrix_normalized = normalize_matrix(df_matrix) 
    df_matrix_normalized.columns = [clean_title_for_match(col) for col in df_matrix_normalized.columns]
    
    print(f"Steam 메타데이터 통합 완료. 크기: {df_meta_steam.shape}")
    
    return df_meta_steam, df_matrix_normalized


# --- 2. 메인 실행 로직 ---

def run_steam_hybrid_test():
    
    print("===============================================================")
    print("          Hybrid 모델 (OpenCritic -> Steam) 테스트 시작")
    print("===============================================================")
    
    # 1. OpenCritic 모델 학습
    print("\n--- 1. OpenCritic 기반 CBF/CF 모델 학습 ---")
    R_pred_df, svd_model_oc = build_cf_model(PATH_CF_DATA) # SVD 모델 객체 이름을 svd_model_oc로 변경
    oc_meta_aligned, cosine_sim_oc = build_cbf_model(PATH_CF_DATA, PATH_META_DATA_OC) 

    # 2. Steam 데이터 전처리 및 준비
    df_meta_steam, df_matrix_normalized = preprocess_steam_data(
        PATH_META_DATA_STEAM, PATH_REVIEW_DATA_STEAM, PATH_USER_MATRIX_STEAM
    )
    
    if df_meta_steam is None or df_matrix_normalized is None:
        print("\n [종료] Steam 데이터 전처리 중 치명적인 오류가 발생하여 작업을 중단합니다.")
        return
    
    # 3. OpenCritic과 Steam 메타데이터 통합
    df_meta_oc_for_concat = oc_meta_aligned.rename(columns={'Title': 'Title', 'Genres': 'Genres', 'Description': 'Description'})[['Title', 'Genres', 'Description', 'Source']]
    df_meta_steam_for_concat = df_meta_steam[['Title', 'Genres', 'Description', 'Source']].copy()

    df_meta_integrated = pd.concat([df_meta_steam_for_concat, df_meta_oc_for_concat], ignore_index=True).drop_duplicates(subset=['Title'], keep='first')
    df_meta_integrated.reset_index(drop=True, inplace=True)
    print(f"통합 메타데이터셋 크기: {df_meta_integrated.shape[0]}개 게임")


    # --- 4. K-Means 튜닝 및 최종 군집화 ---
    print("\n--- 4. K-Means 튜닝 및 최종 군집화 ---")

    df_meta_integrated['content'] = (df_meta_integrated['Genres'] + ' ') * 3 + df_meta_integrated['Description']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix_integrated = tfidf.fit_transform(df_meta_integrated['content'])

    # 4.2 최종 군집화 및 군집 Affinity 계산 (모듈 사용)
    df_meta_integrated, cluster_affinity_score_map = apply_clustering_and_calculate_affinity(
        df_meta_integrated, R_pred_df, FINAL_N_CLUSTERS
    )


    # --- 5. Steam 매트릭스 분리 및 CF 학습 ---
    print(f"\n--- 5. Steam 매트릭스 분리 및 CF 학습 ---")
    
    # R_steam_long 생성 전에 user_id를 문자열로 변환 (타입 불일치 오류 해결)
    df_matrix_normalized.index = df_matrix_normalized.index.astype(str)
    
    R_steam_long = df_matrix_normalized.stack().reset_index()
    R_steam_long.columns = ['user_id', 'Title', 'Rating']
    
    R_train_long, R_test_long = train_test_split(
        R_steam_long, 
        test_size=TEST_SET_RATIO, 
        random_state=42, 
        stratify=R_steam_long['user_id']
    )
    
    R_train_matrix = R_train_long.pivot_table(index='user_id', columns='Title', values='Rating')
    R_test_matrix = R_test_long.pivot_table(index='user_id', columns='Title', values='Rating')
    
    print("  > Steam CF (SVD) 모델 학습 중...")
    
    global_mean_affinity = R_pred_df.values.mean()
    R_train_filled = R_train_matrix.fillna(global_mean_affinity) 
    
    steam_svd = TruncatedSVD(n_components=100, random_state=42)
    steam_svd.fit(R_train_filled)
    
    U_steam = steam_svd.transform(R_train_filled)
    V_T_steam = steam_svd.components_
    
    print(f"  > Steam CF 학습 완료. 사용자 수: {U_steam.shape[0]}, 잠재 요인: {U_steam.shape[1]}")


    # --- 6. Hybrid 예측: Affinity & Steam CF 결합 ---
    print("\n--- 6. Hybrid 예측: Affinity & Steam CF 결합 ---")
    
    # 6.1 OpenCritic Item Vector (V_T) 추출
    V_T_oc_df = pd.DataFrame(svd_model_oc.components_).T
    V_T_oc_df.index = R_pred_df.columns
    V_T_oc_df.index.name = 'Title'

    # 6.2 군집 잠재 요인 벡터 계산 (Cluster Affinity Structure)
    df_oc_games_with_cluster = df_meta_integrated[df_meta_integrated['Source'] == 'OpenCritic'].set_index('Title')
    df_oc_games_with_vector = V_T_oc_df.merge(df_oc_games_with_cluster[['Cluster']], left_index=True, right_index=True, how='inner')
    cluster_mean_vector_df = df_oc_games_with_vector.groupby('Cluster').mean()
    
    hybrid_predictions = [] 
    
    # 6.3 예측 수행: Steam Test Set의 각 (User, Item) 쌍에 대해 예측
    
    for user_id, title, actual_rating in R_test_long.values:
        
        # 1. Base Affinity (OpenCritic 군집 Affinity)
        cluster_id_list = df_meta_integrated[df_meta_integrated['Title'] == title]['Cluster'].tolist()
        cluster_id = cluster_id_list[0] if cluster_id_list else None
        
        if cluster_id in cluster_mean_vector_df.index:
            v_oc_item_vector = cluster_mean_vector_df.loc[cluster_id].values 
            
            # OpenCritic 평론가들이 이 게임에 내릴 평균 예측 점수 (베이스 예측값)
            U_oc_mean = svd_model_oc.transform(R_pred_df.fillna(0)).mean(axis=0)
            A_OC = np.dot(U_oc_mean, v_oc_item_vector)
        else:
            A_OC = global_mean_affinity
            
        # 2. Steam CF Term (개인화)
        try:
            # Steam 훈련 세트에서 사용자 U와 게임 I의 잠재 벡터 추출
            user_idx = R_train_matrix.index.get_loc(user_id) 
            item_idx = R_train_matrix.columns.get_loc(title) 
            
            u_vec = U_steam[user_idx, :]
            v_vec = V_T_steam[:, item_idx]
            
            CF_Term = np.dot(u_vec, v_vec)
            
            # 최종 예측: A_OC (Base) + CF_Term (개인화)
            predicted_rating = A_OC + CF_Term 
            
        except KeyError:
            # Test Set의 사용자/게임이 Train Set에 없을 경우
            predicted_rating = A_OC # 개인화 없이 OC Affinity만 사용

        hybrid_predictions.append({
            'user_id': user_id,
            'Title': title,
            'Rating': actual_rating,
            'Predicted_Rating': predicted_rating
        })

    R_test_long_merged = pd.DataFrame(hybrid_predictions)
    
    
    # --- 7. 예측 정확도 평가 (RMSE) ---
    print("\n--- 7. 최종 예측 정확도 평가 (RMSE) ---")
    
    y_true = R_test_long_merged['Rating'].values
    y_pred = R_test_long_merged['Predicted_Rating'].values
    
    if len(y_true) > 0:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"✅ Hybrid 모델의 Steam Test Set RMSE: {rmse:.4f}")
    else:
        print(" [경고] Test Set에 평가된 데이터가 없어 RMSE 계산을 건너뜁니다.")


    # --- 8. 결과 저장 ---
    # Hybrid Prediction Map 생성 (사용자-아이템 쌍을 Game Title-Avg Score로 저장)
    R_pred_steam_df_long = R_test_long_merged[['Title', 'Predicted_Rating']].copy()
    R_pred_steam_df = R_pred_steam_df_long.groupby('Title')['Predicted_Rating'].mean().rename('Hybrid_Predicted_Score').reset_index()
    
    R_pred_steam_df.to_csv(RESULTS_DIR + 'hybrid_steam_predictions_final.csv', index=False, encoding='utf-8-sig')

    print("\nHybrid 모델 테스트 및 예측 완료: 'hybrid_steam_predictions_final.csv' 파일 확인")
    print("===============================================================")

if __name__ == '__main__':
    run_steam_hybrid_test()