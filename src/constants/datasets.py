from collections import namedtuple

# BASE_PATH = '/Users/soroush/Desktop/University/Projects/XDI/data'

DATASET_CONSTANTS = namedtuple('DATASET_CONSTANTS', 'VALUE')

BUY_DATASET_CONSTANTS = DATASET_CONSTANTS(
    VALUE={
        'NAME': 'buy',
        'REL_PATH': 'data_imputation/buy/buy.csv',
        'ALL_COLUMNS': ['name', 'description', 'manufacturer'],
        'KEY': ['manufacturer'],
        'REST': ['name', 'description']
    }
)

RESTAURANT_DATASET_CONSTANTS = DATASET_CONSTANTS(
    VALUE={
        'NAME': 'restaurant',
        'REL_PATH': 'data_imputation/restaurant/restaurant.csv',
        'ALL_COLUMNS': ['name', 'addr', 'phone', 'type', 'city'],
        'KEY': ['city'],
        'REST': ['name', 'addr', 'phone', 'type']
    }
)

FLIPKART_DATASET_CONSTANTS = DATASET_CONSTANTS(
    VALUE={
        'NAME': 'flipkart',
        'REL_PATH': 'data_imputation/flipkart/flipkart.csv',
        'ALL_COLUMNS': ['uniq_id', 'crawl_timestamp', 'product_url', 'product_name', 'product_category_tree', 'pid', 'retail_price', 'discounted_price', 'image', 'is_FK_Advantage_product', 'description', 'product_rating', 'overall_rating', 'product_specifications', 'brand'],
        'KEY': ['brand'],
        'REST': ['uniq_id', 'crawl_timestamp', 'product_url', 'product_name', 'product_category_tree', 'pid', 'retail_price', 'discounted_price', 'image', 'is_FK_Advantage_product', 'description', 'product_rating', 'overall_rating', 'product_specifications']
    }
)

ZOMATO_DATASET_CONSTANTS = DATASET_CONSTANTS(
    VALUE={
        'NAME': 'zomato',
        'REL_PATH': 'data_imputation/zomato/zomato.csv',
        'ALL_COLUMNS': ['url', 'address', 'name', 'online_order', 'book_table', 'rate', 'votes', 'phone', 'location', 'rest_type', 'dish_liked', 'cuisines', 'approx_cost(for two people)', 'listed_in(type)', 'listed_in(city)'], # 'reviews_list','menu_item', 
        'KEY': ['location'],
        'REST': ['url', 'address', 'name', 'online_order', 'book_table', 'rate', 'votes', 'phone', 'rest_type', 'dish_liked', 'cuisines', 'approx_cost(for two people)', 'listed_in(type)', 'listed_in(city)'] # 'reviews_list','menu_item', 
    }
)


dependency_level = {
    'buy': {'name':0.91, 'description':0.46},
    'restaurant': {'name':0.31, 'addr':0.68, 'phone':0.77, 'type':0.08},
    'flipkart': {'uniq_id':0, 'crawl_timestamp':0.01, 'product_url':0.84, 'product_name':0.78, 'product_category_tree':0.60, 'pid':0, 'retail_price':0.01, 'discounted_price':0, 'image':0.58, 'is_FK_Advantage_product':0, 'description':0.74, 'product_rating':0, 'overall_rating':0, 'product_specifications':0.71},
    'zomato': {'url':0.79, 'address':0.87, 'name':0.03, 'online_order':0, 'book_table':0, 'rate':0, 'votes':0, 'phone':0.03, 'rest_type':0.04, 'dish_liked':0.04, 'cuisines':0.03, 'approx_cost(for two people)':0.03, 'reviews_list':None, 'menu_item':0.07, 'listed_in(type)':0.05, 'listed_in(city)':0.34}
}