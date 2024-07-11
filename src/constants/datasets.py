from collections import namedtuple

# BASE_PATH = '/Users/soroush/Desktop/University/Projects/XDI/data'

DATASET_CONSTANTS = namedtuple('DATASET_CONSTANTS', 'VALUE')

BUY_DATASET_CONSTANTS = DATASET_CONSTANTS(
    VALUE={
        'NAME': 'buy',
        'REL_PATH': 'data_imputation/buy/buy.csv',
        'ALL_COLUMNS': ['name', 'description', 'manufacturer'],
        'KEY': ['manufacturer'],
        'REST': ['name', 'description'],
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
