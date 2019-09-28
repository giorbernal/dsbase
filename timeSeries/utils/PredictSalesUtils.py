# Group of support methods for Predict - sales process

import numpy as np
import pandas as pd
import featuretools as ft

############ Datasets Functions #########################################

def getTestEnriched(testFile, itemsFile):
	test_df = pd.read_csv('../datasets/predict-sales/test.csv')
	items_df = pd.read_csv('../datasets/predict-sales/items.csv')

	items_df.drop(labels=['item_name'], inplace=True, axis=1)

	test_df_enriched = test_df.merge(right=items_df, on='item_id', how='left')
	test_df_enriched['ID_pair'] = test_df_enriched[['shop_id','item_id']].apply(concatAttr, axis=1)
	test_df_enriched['ID_CAT_pair'] = test_df_enriched[['shop_id','item_category_id']].apply(concatAttr, axis=1)
	return test_df_enriched

############ Commons functions #########################################

def concatAttr(x):
    return (str(x[0]) + '-' + str(x[1]))

def getCatAgg(sales_months_df):
    es = ft.EntitySet(id="prediction_sales")
    es = es.entity_from_dataframe(entity_id='sales',dataframe=sales_months_df, index='index')
    es = es.normalize_entity(base_entity_id='sales',
                         new_entity_id='idsCat',
                         index='ID_CAT_pair')
    feature_matrix_idsCat, feature_defs_idsCat = ft.dfs(entityset=es, target_entity='idsCat')
    idsCat = feature_matrix_idsCat.reset_index()
    idsCat_agg = idsCat[['ID_CAT_pair','SUM(sales.item_cnt_day)',
                     'MEAN(sales.item_cnt_day)','MEAN(sales.item_price)',
                     'STD(sales.item_cnt_day)','STD(sales.item_price)',
                     'MAX(sales.item_cnt_day)','MAX(sales.item_price)',
                     'MIN(sales.item_cnt_day)','MIN(sales.item_price)',
                     'SKEW(sales.item_cnt_day)','SKEW(sales.item_price)'
                    ]]
    idsCat_agg.columns = ['ID_CAT_pair','sum_shop_cat_sales',
                      'mean_shop_cat_day','mean_shop_cat_item_price',
                      'std_shop_cat_day','std_shop_cat_item_price',
                      'max_shop_cat_day','max_shop_cat_item_price',
                      'min_shop_cat_day','min_shop_cat_item_price',
                      'skew_shop_cat_day','skew_shop_cat_item_price',
                     ]
    return idsCat_agg

def getItemAgg(sales_months_df):
    es = ft.EntitySet(id="prediction_sales")
    es = es.entity_from_dataframe(entity_id='sales',dataframe=sales_months_df, index='index')
    es = es.normalize_entity(base_entity_id='sales',
                         new_entity_id='ids',
                         index='ID_pair',
                         additional_variables=['ID_CAT_pair'])
    feature_matrix_ids, feature_defs_ids = ft.dfs(entityset=es, target_entity='ids')
    ids = feature_matrix_ids.reset_index()
    return ids

############ Training functions #########################################

# TODO add getTarget for training

# TODO join Traininf three parts

############ Evaluation functions #########################################

def joinEvaluationThreeParts(test, ids, idsCat):
    df = test.merge(right=ids,on='ID_pair',how='left')
    
    # Adapting ID_CAT_pair
    df.drop(labels=['ID_CAT_pair_y'], inplace=True, axis=1)
    columns = np.array(df.columns)
    columns[5]='ID_CAT_pair'
    df.columns = columns
    
    df_completed = df.merge(right=idsCat,on='ID_CAT_pair',how='left')
    
    df_completed_sorted = df_completed[['ID_pair','SUM(sales.item_price)','SUM(sales.item_cnt_day)','STD(sales.item_price)','STD(sales.item_cnt_day)','MAX(sales.item_price)','MAX(sales.item_cnt_day)','SKEW(sales.item_price)','SKEW(sales.item_cnt_day)','MIN(sales.item_price)','MIN(sales.item_cnt_day)','MEAN(sales.item_price)','MEAN(sales.item_cnt_day)','COUNT(sales)','sum_shop_cat_sales','mean_shop_cat_day','mean_shop_cat_item_price','std_shop_cat_day','std_shop_cat_item_price','max_shop_cat_day','max_shop_cat_item_price','min_shop_cat_day','min_shop_cat_item_price','skew_shop_cat_day','skew_shop_cat_item_price','ID_CAT_pair']]
    df_completed_sorted.drop(labels=['ID_CAT_pair'], inplace=True, axis=1)
    return df_completed_sorted

def generateFeaturesForEvaluation(sales_df, months_feature, test_df):
    print('evaluation features window:',months_feature)
    sales_months_df = sales_df[sales_df['date_block_num'].isin(months_feature)]
    sales_months_df.drop(labels=['date_block_num','shop_id','item_id','item_category_id'], inplace=True, axis=1)
    
    idsCat = getCatAgg(sales_months_df)
    ids = getItemAgg(sales_months_df)
    
    joined = joinEvaluationThreeParts(test_df, ids, idsCat)
    
    # Insert the slot component for correlation purposes
    joined['slot'] = joined['COUNT(sales)'].apply(lambda x: months_feature[-1])
    return joined
