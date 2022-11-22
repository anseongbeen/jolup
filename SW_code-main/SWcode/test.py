import sys
import random
from math import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pit
import seaborn as sns
pit.rcParams['figure.figsize'] = [10, 8]

# 사용자 데이터
user_data = pd.read_csv('user.csv', encoding='cp949')
# 평점 데이터
score_data = pd.read_csv('score.csv', encoding='cp949') 
# 음식점 데이터
product_data = pd.read_csv('product.csv', encoding='utf-8')
# 비선호 가게 데이터
dislike_data = pd.read_csv('dislike.csv')


# 필요없는 데이터 삭제
product_data.drop('title', axis = 1, inplace = True)
user_data.drop('email', axis = 1, inplace = True)
user_data.drop('name', axis = 1, inplace = True)

# 컬럼명 변경
user_data.rename(columns={'uCounter':'userId'}, inplace=True) 
product_data.rename(columns={'pCounter':'productId'}, inplace=True)
dislike_data.rename(columns={'productId':'dProduct'}, inplace=True) 
dislike_data.rename(columns={'userId':'dUser'}, inplace=True) 

# nan값이 있는 행 제거
real =dislike_data.dropna(subset=['dProduct']) 


# 데이터 병합
user_product = pd.merge(user_data, score_data,  on = "userId")
user_product_star = pd.merge( product_data, user_product, how='outer', on = "productId")
user_product_star = user_product_star[['userId', 'productId', 'stars']]

# 피벗 테이블 만들기
user_product_star = user_product_star.pivot_table('stars', index = 'userId', columns='productId')


# 빈 값 0으로 채우기
user_product_star.fillna(0, inplace = True)


# 행 열 위치 바꾸기
user_product_star_T = user_product_star.transpose()


# 싫어요 정보 딕셔너리 만들기
subset = real[['dUser', 'dProduct']]
tuples = [tuple(x) for x in subset.to_numpy()]
dislike_dict = {}

for x in tuples:
    dislike_dict.setdefault(x[0], []).append(x[1])

# 싫어요 정보 딕셔너리 버전
# print(dislike_dict)



# 가게별 유사도 구하기
product_sim = cosine_similarity(user_product_star_T, user_product_star_T)
product_sim_df = pd.DataFrame(data=product_sim, index=user_product_star.columns, columns=user_product_star.columns)


# 개인화된 가게별 유사도
def predict_rating_topsim(ratings_arr, product_sim_arr, n=3):

    # 사용자-가게 평점 행렬 크기만큼 0으로 채운 예측 행렬 초기화
    pred = np.zeros(ratings_arr.shape)
    
    # 사용자-가게 평점 행렬의 열 크기만큼 루프 수행.
    for col in range(ratings_arr.shape[1]):
        
        # 유사도 행렬에서 유사도가 큰 순으로 n개 데이터 행렬의 인덱스 반환
        top_n_products = [np.argsort(product_sim_arr[:, col])[:-n-1:-1]]
        
        # 예측 평점을 계산
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = product_sim_arr[col, :][top_n_products].dot(ratings_arr[row,:][top_n_products].T)
            pred[row, col] /= np.sum(np.abs(product_sim_arr[col, :][top_n_products]))
            
    return pred


# 예측 평점 데이터
ratings_pred = predict_rating_topsim(user_product_star.values, product_sim_df.values, n=3)

 
#예측 평점 데이터를 DataFrame으로 변경
ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index=user_product_star.index, columns=user_product_star.columns)


user_rating_id = user_product_star.loc[2, :]
# print(user_rating_id[ user_rating_id > 0 ].sort_values(ascending=False)[:10])


def get_uneaten_products(ratings_matrix, userId):
    # userId로 입력받은 사용자의 모든 가게 정보를 추출해 Series로 반환함.
    # 반환된 user_ratings은 가게명(title)을 인덱스로 가지는 Series 객체임.
    user_rating = ratings_matrix.loc[userId, :]
    
    # user_rating이 0보다 크면 기존에 먹은 가게. 대상 인덱스를 추출해 list 객체로 만듦.
    already_eaten = user_rating[ user_rating>0 ].index.tolist()
    
    # 모든 가게명을 list 객체로 만듦.
    product_list = ratings_matrix.columns.tolist()
    
    # list comprehension으로 already_eaten에 해당하는 가게는 product_list에서 제외함.
    uneaten_list = [product for product in product_list if product not in already_eaten]
    
    return uneaten_list



user_list = ratings_pred_matrix.index.values.tolist()

all_products_list = product_data.values.tolist()
all_products_list = sum(all_products_list, [])
r_product = random.choice(all_products_list)







def recomm_product_by_userid(userId):
    # 예측 평점 DataFrame에서 사용자 id 인덱스와 unseen_list로 들어온 도서명 칼럼을 추출해 가장 예측 평점이 높은 순으로 정렬함.
    pred_df = ratings_pred_matrix
    top_n=3
    uneaten_list = get_uneaten_products(user_product_star, userId)
    recomm_product = pred_df.loc[userId, uneaten_list].sort_values(ascending=False)[:top_n]
    rp = recomm_product.index.values.tolist()
    if userId not in dislike_dict:
        if userId in user_list:      
            if len(rp) == 0:
                test1 = score_data[(score_data['userId']==userId) & (score_data['stars']>=4)]
                test2 = test1[['productId']]
                test2_list = test2.values.tolist()
                test3 = sum(test2_list, [])
                if len(test3) == 0:
                    return r_product
                else:
                    return random.choice(test3)
            else:
                return random.choice(rp)

        elif userId not in user_list:
            return r_product
   
    elif userId in dislike_dict:
        if userId in user_list:
            dislike_values = dislike_dict.get(userId)
            if len(rp) == 0:
                test1 = score_data[(score_data['userId']==userId) & (score_data['stars']>=4)]
                test2 = test1[['productId']]
                test2_list = test2.values.tolist()
                print(test2_list)
                test3 = sum(test2_list, [])
                if len(test3) == 0:
                    complement = list(set(all_products_list) - set(dislike_values))
                    return complement
                else:
                    complement = list(set(test3) - set(dislike_values))
                    return random.choice(complement)
            
            else:
                complement = list(set(rp)-set(dislike_values))
                return random.choice(complement)   
        else:
            complement = list(set(all_products_list)-set(dislike_values))
            return random.choice(complement)


if __name__ == "__main__":
    print(recomm_product_by_userid(int(sys.argv[1])))