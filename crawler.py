import requests
import pandas as pd
import time
import os
import re

# ----------------------------------------------------------------------
# 1. 설정 및 기본 변수
# ----------------------------------------------------------------------

FINDER_API_URL = "https://backend.metacritic.com/finder/metacritic/web"
OUTPUT_CONTENT_CSV = "metacritic_light_content_metadata.csv"

GAME_LIST_LIMIT = 24  
MAX_PAGES_TO_SCRAPE = 5 
DELAY_TIME_SEC = 2    

HEADERS = {
    'User-Agent': 'Mozilla/50 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# ----------------------------------------------------------------------
# 2. 게임 콘텐츠 수집 함수 (Title, Genres, Description 추출)
# ----------------------------------------------------------------------

def fetch_metacritic_light_content():
    """Metacritic Finder API를 호출하여 Title, Genres, Description만 수집합니다."""
    
    game_metadata_list = []
    total_results = float('inf')
    offset = 0
    
    print("--- Metacritic 콘텐츠 수집 시작 (Title, Genres, Description) ---")

    while offset < (MAX_PAGES_TO_SCRAPE * GAME_LIST_LIMIT) and offset < total_results:
        params = {
            'sortBy': '-metaScore',
            'productType': 'games',
            'releaseYearMin': '1958',
            'releaseYearMax': '2025',
            'offset': offset,
            'limit': GAME_LIST_LIMIT
        }

        try:
            response = requests.get(FINDER_API_URL, headers=HEADERS, params=params, timeout=15)
            response.raise_for_status() 
            data = response.json()
            
            data_container = data.get('data', {})
            
            if offset == 0:
                total_results = data_container.get('totalResults', 0)
                max_collect = min(total_results, MAX_PAGES_TO_SCRAPE * GAME_LIST_LIMIT)
                print(f"✅ 총 {total_results}개 게임 중 최대 {max_collect}개 수집 예정.")

            items = data_container.get('items', [])
            
            for item in items:
                game_title = item.get('title')
                
                description = item.get('description', '')
                genres_raw = item.get('genres', [])
                
                # 장르 리스트 추출 및 쉼표로 연결 (One-Hot Encoding 준비)
                genre_names = ', '.join([g.get('name') for g in genres_raw if g.get('name')])
                
                # Slug는 메타데이터에 필요 없으므로 생략하고, AppID를 사용할 수 없으므로 Title에 집중
                if game_title:
                    game_metadata_list.append({
                        'Game_Title': game_title,
                        'Genres': genre_names,
                        'Description': description,
                    })
                
                if len(game_metadata_list) >= max_collect:
                    break
            
            print(f"➡️ 현재까지 {len(game_metadata_list)}개 게임 데이터 수집 완료 (Offset: {offset})")

            offset += GAME_LIST_LIMIT
            time.sleep(DELAY_TIME_SEC)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 요청 오류 발생 (게임 목록): {e}")
            break
        except Exception as e:
            print(f"❌ 데이터 처리 중 오류 발생 (게임 목록): {e}")
            break
            
    return game_metadata_list

# ----------------------------------------------------------------------
# 3. 메인 실행 (이 부분은 메뉴 시스템 내에서 호출됨)
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # 이 파일은 메인 메뉴 시스템 (full_hybrid_recommender.py) 내에서 호출되어야 합니다.
    # 단독 실행 시 테스트용 코드를 여기에 추가할 수 있습니다.
    pass