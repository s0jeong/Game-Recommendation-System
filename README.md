# 🎮 Game Recommendation System
### Hybrid Model: Collaborative Filtering + Content-Based Filtering

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

> OpenCritic 평론가 데이터와 Steam 사용자 리뷰를 활용한 하이브리드 게임 추천 시스템

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Architecture](#️-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Preprocessing](#-data-preprocessing)
- [Modeling](#-modeling)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Team](#-team)

---

## 🎯 Overview

본 프로젝트는 협업 필터링(CF)과 콘텐츠 기반 필터링(CBF)을 결합한 하이브리드 게임 추천 시스템입니다.

### 주요 목표
- ✅ OpenCritic 평론가 평점을 기반으로 한 협업 필터링
- ✅ 게임 메타데이터(장르, 설명)를 활용한 콘텐츠 기반 필터링
- ✅ Cold Start 문제 해결
- ✅ 도메인 간(OpenCritic ↔ Steam) 추천 성능 평가

### Why Hybrid?
- **CF의 강점**: 사용자 간 협업 패턴, 숨겨진 선호도 발견
- **CBF의 강점**: Cold Start 해결, 콘텐츠 유사성 기반 추천
- **하이브리드**: 두 모델의 약점을 서로 보완하여 강건한 추천 시스템 구축

---

## ✨ Features

### 1. Hybrid Recommendation Engine
- **CF (Collaborative Filtering)**: SVD 기반 잠재 요인 분해 (100차원)
- **CBF (Content-Based Filtering)**: TF-IDF + Cosine Similarity
- **가중 결합**: Alpha 파라미터로 CF/CBF 비율 조절
  - `α = 0.5`: 균형 (기본값)
  - `α = 1.0`: CF 100%
  - `α = 0.0`: CBF 100%

### 2. Cold Start Solution
- Steam 리뷰 텍스트 기반 독립 CBF 모델 구축
- Train 데이터에 없는 신규 게임도 즉시 추천 가능
- 804개 Steam 게임 대상 리뷰 기반 유사도 계산

### 3. Robust Data Synchronization
- 텍스트 정규화로 CF-CBF 게임 목록 100% 일치
- 996개 게임, 446명 평론가 데이터 완벽 동기화
- 결측치 전략적 처리 (CF: 0, CBF: '')

---

## 📊 Dataset

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| **Train (CF)** | OpenCritic | 446 × 996 | 평론가 평점 행렬 (0-100점) |
| **Content (CBF)** | Metacritic | 996 games | 게임 메타데이터 (장르, 설명) |
| **Test (Ratings)** | Steam | 804 games | Steam 사용자 평점 |
| **Test (Reviews)** | Steam | 804 games | Steam 리뷰 텍스트 |

### 데이터 특성
- **OpenCritic**: 평론가들의 전문적인 평가 (엄격한 기준)
  - 평균 평점: ~9점 (낮음)
- **Steam**: 일반 사용자 리뷰 (관대한 경향)
  - 평균 평점: ~74점 (높음)
- **도메인 갭**: RMSE 65.07 (두 플랫폼 간 평가 기준 차이)
- **콘텐츠 정보**: 996개 중 449개(45%)만 메타데이터 보유

### 공통 게임
OpenCritic과 Steam에 모두 존재하는 게임:
- Hades II
- Dying Light: The Beast
- Hollow Knight: Silksong
- Cronos: The New Dawn

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Input Data                          │
│  ┌──────────────────┐         ┌──────────────────┐      │
│  │   CF Data        │         │   CBF Data       │      │
│  │   (446×996)      │         │   (Metadata)     │      │
│  │  Rating Matrix   │         │  Genre + Desc    │      │
│  └──────────────────┘         └──────────────────┘      │
│         │                              │                │
│         ↓                              ↓                │
│  ┌──────────────────┐         ┌──────────────────┐      │
│  │  Preprocessing   │         │  Preprocessing   │      │
│  │  - fillna(0)     │         │  - fillna('')    │      │
│  │  - Normalize     │         │  - Genre 3x      │      │
│  └──────────────────┘         └──────────────────┘      │
│         │                              │                │
│         ↓                              ↓                │
│  ┌──────────────────┐         ┌──────────────────┐      │
│  │      SVD         │         │     TF-IDF       │      │
│  │  (100 factors)   │         │   + Cosine Sim   │      │
│  └──────────────────┘         └──────────────────┘      │
│         │                              │                │
│         ↓                              ↓                │
│    CF Score (446×996)          CBF Score (996×996)      │
│         │                              │                │
│         └──────────────┬───────────────┘                │
│                        ↓                                │
│              ┌────────────────────┐                     │
│              │  MinMax Normalize  │                     │
│              │  CF_norm, CBF_norm │                     │
│              └────────────────────┘                     │
│                        ↓                                │
│              ┌────────────────────┐                     │
│              │  Weighted Combine  │                     │
│              │  α·CF + (1-α)·CBF  │                     │
│              └────────────────────┘                     │
│                        ↓                                │
│              ┌────────────────────┐                     │
│              │   Hybrid Score     │                     │
│              │   Top-N Ranking    │                     │
│              └────────────────────┘                     │
│                        ↓                                │
│              ┌────────────────────┐                     │
│              │  Recommendations   │                     │
│              └────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Requirements
```
Python 3.8+
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
joblib >= 1.0.0
```

### Setup
```bash
# 1. Clone repository
git clone https://github.com/yourusername/game-recommendation.git
cd game-recommendation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets (place in data/ folder)
# Required files:
# - density_processed.csv
# - metacritic_light_content_metadata.csv
# - normalized_preprocessed_steam_matrix.csv
# - review_data.csv
# - game_metadata_processed.csv
```

### requirements.txt
```
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
joblib==1.1.0
```

---

## 💻 Usage

### 1. Run Full Pipeline
전체 파이프라인 실행 (전처리 → 모델링 → 평가)
```python
# Run the Jupyter notebook
jupyter notebook CF+CBF.ipynb

# Or run cells sequentially:
# - Cell 2: CBF Model Training
# - Cell 3: CF Model Training
# - Cell 5: Hybrid Model Construction
# - Cell 7: Evaluation
```

### 2. Get Hybrid Recommendations
특정 평론가에게 게임 추천
```python
from hybrid_recommender import get_hybrid_recommendations

# 예시: '1UP' 평론가에게 추천
recommendations = get_hybrid_recommendations(
    critic_name='1UP',
    alpha=0.5,      # CF:CBF = 50:50
    top_n=10        # 상위 10개 게임
)

print(recommendations)
```

**출력 예시**:
```
                         Game_Title  CF_Score  CBF_Score  Hybrid_Score
0       Call of Duty: World at War     35.13      0.067         0.647
1           Rise of Nations (2003)     27.07      0.067         0.612
2                       Homeworld 2     17.16      0.064         0.552
3  Starcraft II: Heart of the Swarm      4.37      0.071         0.549
4         Age of Mythology: Retold      2.22      0.070         0.532
```

### 3. Cold Start Test
Train 데이터에 없는 신규 게임 추천
```python
from cold_start_cbf import get_steam_cbf_recommendations

# OpenCritic에 없는 Steam 게임
new_game_recs = get_steam_cbf_recommendations(
    title='BLACK SOULS',
    top_n=5
)

print(new_game_recs)
```

**출력 예시**:
```
1. SILENT HILL f
2. 로스트 아이돌론스: 베일 오브 더 위치
3. Hades II
4. BALL x PIT
5. No, I'm not a Human
```

### 4. Adjust Alpha Parameter
CF/CBF 비율 조정
```python
# CF 중심 (협업 필터링 강조)
cf_heavy = get_hybrid_recommendations('1UP', alpha=0.8, top_n=10)

# CBF 중심 (콘텐츠 유사성 강조)
cbf_heavy = get_hybrid_recommendations('1UP', alpha=0.2, top_n=10)

# 균형 (기본)
balanced = get_hybrid_recommendations('1UP', alpha=0.5, top_n=10)
```

---

## 🔧 Data Preprocessing

### Overview
전처리는 6단계 파이프라인으로 구성됩니다:

```
데이터 로드 → 텍스트 정규화 → 데이터 동기화 → 결측치 처리 → 특성 공학 → Test 데이터 준비
```

### Step 1: Data Loading
```python
# CF 데이터 (평점 행렬)
cf_data = pd.read_csv('density_processed.csv', index_col=0)
# Shape: (446 critics, 996 games)

# CBF 데이터 (메타데이터)
meta_data = pd.read_csv('metacritic_light_content_metadata.csv')
# Columns: Game_Title, Genres, Description
```

### Step 2: Text Normalization
게임 이름의 특수 문자와 공백을 정규화하여 데이터 일관성 확보
```python
def clean_title_for_match(title):
    # 특수 공백 제거 (\xa0, \u200b)
    cleaned = str(title).replace(u'\xa0', u' ').replace(u'\u200b', u' ')
    # 줄바꿈 제거 (\r, \n)
    cleaned = cleaned.replace('\r', ' ').replace('\n', ' ').strip()
    # 연속 공백을 하나로
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

# 모든 데이터셋에 적용
cf_data.columns = [clean_title_for_match(col) for col in cf_data.columns]
meta_data['Game_Title'] = meta_data['Game_Title'].apply(clean_title_for_match)
```

**효과**: "Hades  II" vs "Hades\xa0II" → "Hades II" (통일)

### Step 3: Data Synchronization
CF와 CBF의 게임 목록을 정확히 일치시킴 (996개 게임)
```python
# 1. CF의 996개 게임을 기준으로 설정
games_in_cf = cf_data.columns

# 2. CBF 메타데이터 중복 제거
meta_data_unique = meta_data.drop_duplicates(subset=['Game_Title'], keep='first')

# 3. CF 순서로 CBF 재정렬
meta_data_indexed = meta_data_unique.set_index('Game_Title')
meta_data_aligned = meta_data_indexed.reindex(games_in_cf)

# 4. 결측치 처리
meta_data_aligned.fillna('', inplace=True)
```

**결과**: 
- CF와 CBF가 동일한 996개 게임, 동일한 순서 보유
- 547개 게임은 콘텐츠 정보 없음 ('')

### Step 4: Missing Value Handling
각 모델의 알고리즘 특성에 맞게 결측치 처리
```python
# CF: NaN → 0 (SVD는 완전한 행렬 필요)
cf_data_filled = cf_data.fillna(0)
# 0 = "평가 안 함" ≠ "낮은 평점"

# CBF: NaN → '' (TF-IDF는 빈 문자열을 zero vector로 처리)
meta_data_aligned.fillna('', inplace=True)
# 콘텐츠 없어도 CF 점수로 추천 가능 (하이브리드 장점)
```

### Step 5: Feature Engineering
장르에 3배 가중치를 부여하여 핵심 특성 강조
```python
# 장르를 3번 반복 + 설명 1번
meta_data_aligned['content'] = (
    (meta_data_aligned['Genres'] + ' ') * 3 + 
    meta_data_aligned['Description']
)
```

**예시**:
- Input: `Genres="Action RPG"`, `Description="Great game"`
- Output: `"Action RPG Action RPG Action RPG Great game"`
- 효과: TF-IDF 계산 시 장르 단어가 3배 더 중요하게 인식됨

### Step 6: Test Data Preparation
Steam 데이터 전처리 및 Cold Start용 CBF 모델 구축
```python
# 1. AppID별 리뷰 통합
game_reviews = steam_reviews.groupby('AppID')['review_text'].apply(' '.join)

# 2. 메타데이터와 병합
cbf_test = pd.merge(game_reviews, steam_meta[['AppID', 'name']])

# 3. TF-IDF 벡터화 (804개 게임)
tfidf_test = TfidfVectorizer(max_features=5000)
tfidf_matrix_test = tfidf_test.fit_transform(cbf_test['review_text'])

# 4. 코사인 유사도 계산
cosine_sim_test = cosine_similarity(tfidf_matrix_test)  # (804×804)
```

---

## 🤖 Modeling

### 1. CF Model: SVD (Collaborative Filtering)
잠재 요인 분해를 통한 협업 필터링
```python
from sklearn.decomposition import TruncatedSVD

# SVD 모델 초기화
svd = TruncatedSVD(
    n_components=100,    # 100개 잠재 요인
    random_state=42      # 재현성
)

# 학습
svd.fit(cf_data_filled)  # (446, 996)

# 예측 평점 행렬 생성
U = svd.transform(cf_data_filled)  # (446, 100)
V_T = svd.components_               # (100, 996)
R_pred = np.dot(U, V_T)            # (446, 996)
```

**핵심 아이디어**:
- 446×996 평점 행렬을 100차원으로 압축
- 사용자와 게임의 잠재적 특성(장르 선호, 난이도 등) 학습
- 평가하지 않은 게임의 점수 예측

### 2. CBF Model: TF-IDF + Cosine Similarity
콘텐츠 기반 게임 유사도 계산
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF 벡터화
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(meta_data_aligned['content'])
# Output: (996, n_features) sparse matrix

# 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Output: (996, 996) similarity matrix
```

**핵심 아이디어**:
- 게임의 장르와 설명을 TF-IDF로 벡터화
- 게임 간 코사인 유사도 계산 (0~1)
- 사용자가 좋아한 게임과 비슷한 게임 추천

### 3. Hybrid Model: Weighted Combination
CF와 CBF 점수를 정규화 후 가중 결합
```python
from sklearn.preprocessing import MinMaxScaler

def get_hybrid_recommendations(critic_name, alpha=0.5, top_n=10):
    # 1. CF 점수 추출
    cf_scores = R_pred_df.loc[critic_name]
    
    # 2. CBF 점수 계산
    original_ratings = cf_data_original.loc[critic_name]
    liked_games = original_ratings[original_ratings >= 80].index
    
    cbf_scores = []
    for game_idx in range(996):
        # 좋아한 게임들과의 평균 유사도
        avg_sim = np.mean([cosine_sim[game_idx, liked_idx] 
                          for liked_idx in liked_indices])
        cbf_scores.append(avg_sim)
    
    # 3. MinMax 정규화 (0~1)
    scaler = MinMaxScaler()
    cf_norm = scaler.fit_transform(cf_scores.values.reshape(-1, 1))
    cbf_norm = scaler.fit_transform(np.array(cbf_scores).reshape(-1, 1))
    
    # 4. 가중 결합
    hybrid_score = alpha * cf_norm + (1 - alpha) * cbf_norm
    
    # 5. 이미 평가한 게임 제외 후 정렬
    # ...
    
    return top_n_games
```

**핵심 아이디어**:
- CF와 CBF 점수를 0~1로 정규화하여 공정한 비교
- Alpha 파라미터로 두 모델의 영향력 조절
- 콘텐츠 없는 547개 게임은 CBF=0이므로 자동으로 CF 100% 반영

---

## 📈 Evaluation

### Task 1: Domain Gap Analysis (RMSE)
OpenCritic과 Steam의 4개 공통 게임으로 도메인 간 차이 측정

| Game | OpenCritic Pred | Steam Actual | Gap |
|------|-----------------|--------------|-----|
| Hades II | 8.87 | 81.24 | **-72.37** |
| Dying Light: The Beast | 13.01 | 69.00 | **-55.99** |
| Hollow Knight: Silksong | 8.09 | 72.04 | **-63.95** |
| Cronos: The New Dawn | 7.53 | 74.42 | **-66.89** |

**RMSE: 65.07**

#### 해석
- **평론가(OpenCritic)**: 전문적이고 엄격한 평가 기준 (평균 ~9점)
- **사용자(Steam)**: 일반 대중의 관대한 평가 (평균 ~74점)
- **도메인 갭**: 약 65점 차이 → 두 플랫폼의 평가 기준이 근본적으로 다름
- **시사점**: 단순 전이 학습 어려움 → Domain Adaptation 기법 필요

### Task 2: Cold Start Test
Train 데이터(OpenCritic)에 없는 게임도 추천 가능한지 테스트

#### 테스트 게임
**BLACK SOULS** (OpenCritic에 없는 Steam 전용 게임)

#### 추천 결과 ✅
```
1. SILENT HILL f
2. 로스트 아이돌론스: 베일 오브 더 위치
3. Hades II
4. BALL x PIT
5. No, I'm not a Human
```

#### 성공 요인
- ✅ **Steam 리뷰 기반 독립 CBF 모델**: 804개 게임의 리뷰 텍스트로 TF-IDF 학습
- ✅ **CF 점수 불필요**: CBF만으로도 유사 게임 발견 가능
- ✅ **장르 유사성**: 호러/다크 판타지 RPG 게임들이 정확히 추천됨

#### 의의
- 신규 출시 게임도 **즉시 추천 가능**
- Train 데이터에 의존하지 않는 **독립적 추천 시스템**
- **실용적 가치** 높음 (실제 서비스 적용 가능)

---

## 🎯 Results

### ✅ Successes

| 항목 | 설명 | 성과 |
|------|------|------|
| **하이브리드 시너지** | CF+CBF 결합으로 각 모델의 약점 보완 | 콘텐츠 없는 547개 게임도 CF로 추천 가능 |
| **Cold Start 해결** | Steam CBF로 신규 게임 추천 | BLACK SOULS 테스트 성공 |
| **데이터 동기화** | 텍스트 정규화로 100% 일치 | CF-CBF 완벽 동기화 (996개) |
| **장르 강조** | 3:1 가중치로 핵심 특성 반영 | 유사 장르 게임 정확히 추천 |

### ⚠️ Limitations & Future Work

| 문제점 | 영향 | 개선 방향 |
|--------|------|-----------|
| **도메인 갭** | RMSE 65.07 (큰 차이) | Domain Adaptation, Transfer Learning |
| **데이터 희소성** | 547개(55%) 콘텐츠 없음 | 외부 메타데이터 수집 (IGDB, Rawg) |
| **단순 선형 결합** | Alpha 고정값 사용 | Adaptive Weighting (강화학습, Neural Network) |
| **평가 제한** | 4개 공통 게임만 평가 | 더 많은 공통 데이터 확보 |

### 🚀 Future Improvements

#### 1. Deep Learning 기반 추천
- **Neural Collaborative Filtering**: MLP로 비선형 패턴 학습
- **Wide & Deep**: 암기와 일반화 동시 달성
- **Transformer**: Self-attention으로 게임 간 복잡한 관계 모델링

#### 2. Context-aware Recommendation
- **시간적 맥락**: 출시 연도, 시즌, 트렌드 반영
- **플랫폼 특성**: PC, Console, Mobile 선호도
- **사용자 특성**: 플레이 시간, 선호 장르, 연령대

#### 3. Multi-modal Learning
- **이미지**: 게임 스크린샷, 포스터 → CNN
- **비디오**: 트레일러 영상 → Video Encoder
- **오디오**: BGM, 사운드 이펙트 → Audio Feature

#### 4. Online Learning & A/B Testing
- **실시간 피드백**: 사용자 클릭/구매 데이터로 Alpha 자동 조정
- **A/B Testing**: 여러 Alpha 값 실험하여 최적값 발견
- **Bandit Algorithm**: Explore-Exploit 균형

---

## 📁 Project Structure

```
game-recommendation/
│
├── data/                               # 원본 데이터
│   ├── density_processed.csv          # CF 평점 행렬 (446×996)
│   ├── metacritic_light_content_metadata.csv  # CBF 메타데이터
│   ├── normalized_preprocessed_steam_matrix.csv  # Steam 평점
│   ├── review_data.csv                # Steam 리뷰 텍스트
│   └── game_metadata_processed.csv    # Steam 게임 메타데이터
│
├── results/                            # 모델 결과물
│   ├── R_pred_df.csv                  # CF 예측 평점 (446×996)
│   ├── svd_model.pkl                  # 학습된 SVD 모델
│   ├── cbf_cosine_sim.npy             # CBF 유사도 행렬 (996×996)
│   └── cbf_meta_data_aligned.csv      # 정렬된 메타데이터
│
├── notebooks/                          # Jupyter Notebook
│   └── CF+CBF.ipynb                   # 전체 파이프라인
│
├── requirements.txt                    # 의존성 패키지
├── README.md                           # 프로젝트 설명 (현재 파일)
└── LICENSE                             # 라이선스
```

---

## 👥 Team

**Machine Learning Term Project - Team 11**

| Role | Responsibilities | Contact |
|------|------------------|---------|
| **Data Preprocessing** | 텍스트 정규화, 데이터 동기화, 결측치 처리 | - |
| **CF Modeling** | SVD 학습, 평점 예측 행렬 생성 | - |
| **CBF Modeling** | TF-IDF 벡터화, 코사인 유사도 계산 | - |
| **Hybrid System** | 점수 결합, 하이브리드 추천 알고리즘 | - |

> 모든 팀원이 협력하여 프로젝트를 완성했습니다.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References

### Papers
- Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix factorization techniques for recommender systems*. Computer, 42(8), 30-37.
- Lops, P., De Gemmis, M., & Semeraro, G. (2011). *Content-based recommender systems: State of the art and trends*. Recommender systems handbook, 73-105.

### Datasets
- **OpenCritic**: Professional game critic reviews and ratings
- **Metacritic**: Game metadata (genres, descriptions)
- **Steam**: User reviews and ratings

### Tools & Libraries
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms (SVD, TF-IDF, MinMaxScaler)
- **Joblib**: Model serialization

---

## 🙏 Acknowledgments

- Steam for user review and rating data
- Metacritic for comprehensive game metadata
- All team members for their dedication and collaboration

---

## 📊 Performance Summary

| Metric | Value | Description |
|--------|-------|-------------|
| **Train Data** | 446 × 996 | OpenCritic 평론가 평점 |
| **Test Data** | 804 games | Steam 사용자 리뷰 |
| **SVD Components** | 100 | 잠재 요인 개수 |
| **Content Coverage** | 45% (449/996) | 메타데이터 보유율 |
| **Domain Gap RMSE** | 65.07 | OpenCritic vs Steam |
| **Cold Start** | ✅ Success | BLACK SOULS 추천 성공 |

---

## 🎓 Educational Value

이 프로젝트는 다음을 학습할 수 있습니다:

### 추천 시스템 기초
- ✅ Collaborative Filtering (협업 필터링)
- ✅ Content-Based Filtering (콘텐츠 기반 필터링)
- ✅ Hybrid Approach (하이브리드 접근법)

### 머신러닝 기법
- ✅ SVD (Singular Value Decomposition)
- ✅ TF-IDF (Term Frequency-Inverse Document Frequency)
- ✅ Cosine Similarity
- ✅ MinMax Normalization

### 데이터 처리
- ✅ 텍스트 정규화 (Text Normalization)
- ✅ 데이터 동기화 (Data Synchronization)
- ✅ 결측치 처리 (Missing Value Handling)
- ✅ 특성 공학 (Feature Engineering)

### 평가 방법론
- ✅ RMSE (Root Mean Square Error)
- ✅ Domain Gap Analysis
- ✅ Cold Start Testing

---

## 💡 Key Takeaways

### 1. 하이브리드의 힘
CF와 CBF를 결합하면 각 모델의 약점을 보완할 수 있습니다:
- CF 약점(Cold Start) → CBF로 해결
- CBF 약점(개인화 부족) → CF로 보완

### 2. 데이터 전처리의 중요성
전체 프로젝트 시간의 50% 이상이 전처리에 투입되었습니다:
- 텍스트 정규화로 데이터 일관성 확보
- 동기화로 모델 간 완벽한 정렬

### 3. 도메인 지식 활용
장르에 3배 가중치를 부여한 것은 게임 도메인 지식에 기반:
- 장르가 게임의 핵심 특성을 가장 잘 대표
- 단순 기술적 접근보다 도메인 이해가 중요

### 4. Cold Start 해결의 실용성
신규 게임도 즉시 추천 가능한 시스템이 실제 서비스에서는 필수:
- Steam 리뷰만으로 독립적 추천 가능
- Train 데이터 업데이트 없이도 작동

---

## 🐛 Known Issues

현재 알려진 제한사항과 이슈입니다:

1. **메타데이터 부족**: 996개 중 547개(55%) 게임은 장르/설명 정보 없음
   - 영향: 해당 게임들은 CBF 점수 0 (CF만 의존)
   - 해결: 외부 API (IGDB, Rawg) 활용하여 메타데이터 보강 필요

2. **도메인 갭**: OpenCritic과 Steam의 평가 기준이 매우 다름 (RMSE 65.07)
   - 영향: 직접적인 전이 학습 어려움
   - 해결: Domain Adaptation 기법 (DANN, CORAL) 적용 필요

3. **Alpha 고정값**: 모든 사용자에게 동일한 α=0.5 적용
   - 영향: 개인화 부족
   - 해결: 사용자별 최적 Alpha 학습 (강화학습, Neural Network)

4. **평가 데이터 부족**: 4개 공통 게임으로만 평가
   - 영향: 일반화 성능 검증 제한적
   - 해결: 더 많은 크로스 플랫폼 데이터 수집

---

## 🔄 Update Log

### Version 1.0.0 (2024-11-10)
- ✅ Initial release
- ✅ CF Model (SVD) implementation
- ✅ CBF Model (TF-IDF) implementation
- ✅ Hybrid recommendation system
- ✅ Cold Start solution
- ✅ Domain Gap evaluation
- ✅ Complete documentation

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/game-recommendation&type=Date)](https://star-history.com/#yourusername/game-recommendation&Date)

---

## 📝 Citation

이 프로젝트를 연구나 프로젝트에 사용하신다면 다음과 같이 인용해주세요:

```bibtex
@misc{game-recommendation-2024,
  author = {Team 11},
  title = {Game Recommendation System: Hybrid CF+CBF Approach},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/game-recommendation}}
}
```

<div align="center">

**Made with ❤️ by Team 11**

[⬆ Back to Top](#-game-recommendation-system)

</div>
