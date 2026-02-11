DATASETS = {
    'buy': {
        'NAME': 'buy',
        'REL_PATH': 'data_imputation/buy/buy.csv',
        'ALL_COLUMNS': ['name', 'description', 'manufacturer'],
        'KEY': ['manufacturer']
    },
    'restaurant': {
        'NAME': 'restaurant',
        'REL_PATH': 'data_imputation/restaurant/restaurant.csv',
        'ALL_COLUMNS': ['name', 'addr', 'phone', 'type', 'city'],
        'KEY': ['city']
    },
    'flipkart': {
        'NAME': 'flipkart',
        'REL_PATH': 'data_imputation/flipkart/flipkart.csv',
        'ALL_COLUMNS': ['uniq_id', 'crawl_timestamp', 'product_url', 'product_name', 'product_category_tree', 'pid', 'retail_price', 'discounted_price', 'image', 'is_FK_Advantage_product', 'description', 'product_rating', 'overall_rating', 'product_specifications', 'brand'],
        'KEY': ['brand']
    },
    'zomato': {
        'NAME': 'zomato',
        'REL_PATH': 'data_imputation/zomato/zomato.csv',
        # 'reviews_list','menu_item',
        'ALL_COLUMNS': ['url', 'address', 'name', 'online_order', 'book_table', 'rate', 'votes', 'phone', 'location', 'rest_type', 'dish_liked', 'cuisines', 'approx_cost(for two people)', 'listed_in(type)', 'listed_in(city)'],
        'KEY': ['location']
    },
    'walmart': {
        'NAME': 'walmart',
        'REL_PATH': 'data_imputation/walmart/walmart.csv',
        'ALL_COLUMNS': ['index', 'SHIPPING_LOCATION', 'DEPARTMENT', 'CATEGORY', 'SUBCATEGORY', 'BREADCRUMBS', 'SKU', 'PRODUCT_URL', 'PRODUCT_NAME', 'BRAND', 'PRICE_RETAIL', 'PRICE_CURRENT', 'PRODUCT_SIZE', 'RunDate', 'tid'],
        'KEY': ['BRAND']
    },
    'phone': {
        'NAME': 'phone',
        'REL_PATH': 'data_imputation/phone/phone.csv',
        'ALL_COLUMNS': ['brand', 'model', 'network_technology', '2G_bands', '3G_bands', '4G_bands', 'network_speed', 'GPRS', 'EDGE', 'announced', 'status', 'dimentions', 'weight_g', 'weight_oz', 'SIM', 'display_type', 'display_resolution', 'display_size', 'OS', 'CPU', 'Chipset', 'GPU', 'memory_card', 'internal_memory', 'RAM', 'primary_camera', 'secondary_camera', 'loud_speaker', 'audio_jack', 'WLAN', 'bluetooth', 'GPS', 'NFC', 'radio', 'USB', 'sensors', 'battery', 'colors', 'approx_price_EUR', 'img_url'],
        'KEY': ['brand']
    },
    'phone_2': {
        'NAME': 'phone_2',
        'REL_PATH': 'data_imputation/phone_2/phone_2.csv',
        'ALL_COLUMNS': ['Product Name', 'Brand Name', 'Price', 'Rating', 'Reviews', 'Review Votes'],
        'KEY': ['Brand Name']
    }
}

dependency_level = {
    'buy': {'name': 0.91, 'description': 0.46},
    'restaurant': {'name': 0.31, 'addr': 0.68, 'phone': 0.77, 'type': 0.08},
    'flipkart': {'uniq_id': 0, 'crawl_timestamp': 0.01, 'product_url': 0.84, 'product_name': 0.78, 'product_category_tree': 0.60, 'pid': 0, 'retail_price': 0.01, 'discounted_price': 0, 'image': 0.58, 'is_FK_Advantage_product': 0, 'description': 0.74, 'product_rating': 0, 'overall_rating': 0, 'product_specifications': 0.71},
    'zomato': {'url': 0.79, 'address': 0.87, 'name': 0.03, 'online_order': 0, 'book_table': 0, 'rate': 0, 'votes': 0, 'phone': 0.03, 'rest_type': 0.04, 'dish_liked': 0.04, 'cuisines': 0.03, 'approx_cost(for two people)': 0.03, 'reviews_list': None, 'menu_item': 0.07, 'listed_in(type)': 0.05, 'listed_in(city)': 0.34},
    'walmart': {'index': 0, 'SHIPPING_LOCATION': 0, 'DEPARTMENT': 0, 'CATEGORY': 0.11, 'SUBCATEGORY': 0.03, 'BREADCRUMBS': 0.13, 'SKU': 0, 'PRODUCT_URL': 0.77, 'PRODUCT_NAME': 0.77, 'BRAND': 0, 'PRICE_RETAIL': 0, 'PRICE_CURRENT': 0, 'PRODUCT_SIZE': 0, 'PROMOTION': 0, 'RunDate': 0, 'tid': 0},
    'phone': {},
    'phone_2': {}
}

# Example Usage
# print(DATASETS['buy']['NAME'])  # Output: buy
# print(DATASETS['walmart']['ALL_COLUMNS'])
