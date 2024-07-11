from collections import namedtuple

FIXED_ONE_SHOT_TEXT = 'You are given one example to guide your response: '
FIXED_FEW_SHOT_TEXT = 'You are given some examples to guide your response: '

DATASET_PROMPTS = namedtuple('DATASET_PROMPTS', 'VALUE')

BUY_DATASET_PROMPTS = DATASET_PROMPTS(
    VALUE={
        'FIXED_INITIAL':
        '''
Your task is to generate a simple rule that identifies the product manufacturer based on the given information. Your response should be just the rule in the following format without any additional information. \n <entity-type>:<named-entity>-><manufacturer> 
<named-entity> is the entity that is useful to generate the manufacturer name and <entity-type> is its type ('CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART'). <manufacturer> must be one these values: ["LINKSYS", "Netgear", "Belkin", "LACIE", "Canon", "Kensington", "Tripp Lite", "Sony", "D-Link", "Panasonic", "Logitech", "Sennheiser", "Garmin", "OMNIMOUNT SYSTEMS, INC", "Sanus", "Plantronics Bluetooth", "Peerless", "UNIVERSAL REMOTE CONTROL, INC", "Nikon", "Sirius", "Audiovox", "Xm", "Fellowes", "Uniden", "Pioneer", "BOSE", "Apple", "Sharp", "Techcraft", "Denon", "Toshiba", "Kenwood", "Griffin", "Olympus", "Tech Craft", "TRANSCEND INFORMATION", "Polk Audio", "Tom Tom", "Coby", "Case Logic", "Jabra", "Yamaha", "Monster Cable", "Monster", "Bracketron", "Speck Products", "Klipsch", "Microsoft", "Contour", "Canon Camcorders", "Samsung", "ELGATO SYSTEMS", "Plantronics", "Sony DSLR", "Onkyo", "Alpine", "Z-LINE DESIGNS", "PIONEER ELECTRONICS USA", "Pure Digital Technol", "Boston Acoustics", "Mitsubishi", "LG Electronics"] \n You are given one example to guide your response: 
Nikon [ORG] 55-200mm [QUANTITY] 0.28x - 55mm [QUANTITY] 200mm - f/4 [QUANTITY] 5.6 [CARDINAL], Rule: ORG:Nikon->Nikon\n
''',
        'FIXED_QUERRY': 'Rule: ?'
    }
)

RESTAURANT_DATASET_PROMPTS = DATASET_PROMPTS(
    VALUE={
        'FIXED_INITIAL':
        '''
Your task is to generate a simple rule that identifies the city where the restaurant is located based on the given information. Your response should be just the rule in the following format without any additional information. \n <entity-type>:<named-entity>-><city> 
<named-entity> is the entity that is useful to generate the city name and <entity-type> is its type. \n You are given one example to guide your response: 
{"ORG": ["Sheraton Palace Hotel"], "FAC": ["2 New Montgomery Street"], "GPE": ["Market Street"], "CARDINAL": ["415/546-5000"]}, Rule: CARDINAL:415/546-5000->San Francisco \n
''',
        'FIXED_QUERRY': 'Rule: ?',
        'MAPPING_HANDLER_INITIAL':
        '''
Act as a function named find_city that gets input (sth that identifies a citylike area code, etc.) and returns the city name for the given input. You should just return the output (city name), not anything else. Here are a few examples:

Input: 213/467-1108
Output: los Angeles

Input: 415/399-0499
Output: san francisco

Input: 310/456-0488
Output: malibu

Now return the city name without any other description.
''',
            'MAPPING_HANDLER_QUERRY': 'Output: ?'
    }
)

# Your task is to generate a Python dictionary that shows the named entities in the given text. Your response should be just a Python dictionary where the type of each named entity is the key and the entities of each type are values of that key.

NER_PROMPTS = namedtuple('NER_PROMPTS', 'VALUE')
ALL_MODELS_NER_PROMPTS = NER_PROMPTS(
    VALUE={
        'ANNOTATOR_FIXED_INITIAL':
        '''
Your task is to annotate the named entities in the given text. Your response should be just the annotated version of the input where the type of each named entity is mentioned in a bracket after that. 

Entity types and their description are as follows:
Person: Names of people.
PersonType: Job types or roles held by a person.
Location: Natural and human-made landmarks, structures, geographical features, and geopolitical entities.
Organization: Companies, political groups, musical bands, sports clubs, government bodies, and public organizations. Nationalities and religions are not included in this entity type.
Event: Historical, social, and naturally occurring events (like Cultural, Natural, and Sports events).
Product: Physical objects of various categories.
Skill: A capability, skill, or expertise.
PhoneNumber: Phone numbers.
Email: Email addresses.
URL: URLs to websites.
IP: Network IP addresses.
DateTime: Dates and times of day (like Date, Time, DateRange, TimeRange, Duration, and Set).
Quantity: Numbers and numeric quantities (like Number, Percentage, Age, Currency, Dimensions, and Temperature).

Here is one example:
Input: Nikon 55-200mm 0.28x - 55mm 200mm - f/4 5.6
Output: 
Nikon [Organization] 55-200mm [Quantity] 0.28x - 55mm [Quantity] 200mm - f/4 [Quantity] 5.6 [Quantity]
Input: 
''',
        'ANNOTATOR_FIXED_QUERRY': '''\nOutput: ?''',
        'NER_FIXED_INITIAL': '''
Your task is to generate a list of entity types for the given text in the given text. Your response should be just a a list of unique entity types where the type. 

Entity types and their description are as follows:
Person: Names of people.
PersonType: Job types or roles held by a person.
Location: Natural and human-made landmarks, structures, geographical features, and geopolitical entities.
Organization: Companies, political groups, musical bands, sports clubs, government bodies, and public organizations. Nationalities and religions are not included in this entity type.
Event: Historical, social, and naturally occurring events (like Cultural, Natural, and Sports events).
Product: Physical objects of various categories.
Skill: A capability, skill, or expertise.
PhoneNumber: Phone numbers.
Email: Email addresses.
URL: URLs to websites.
IP: Network IP addresses.
DateTime: Dates and times of day (like Date, Time, DateRange, TimeRange, Duration, and Set).
Quantity: Numbers and numeric quantities (like Number, Percentage, Age, Currency, Dimensions, and Temperature).

Here are three examples:

Input: "212/371-2323 404/525-2062 415-863-8205 (415)863-8205"
Output: ['PhoneNumber']

Input: "san francisco Atlanta new york"
Output: ['Location']

Input: "Nikon 55-200mm 0.28x - 55mm "
Output: ['Organization', 'Quantity']

Now return the list of entity types of the following input without any other description.
Input: 
''',
        'NER_FIXED_QUERRY': '''\nOutput: ?''',
        'NER_DIC_FIXED_INITIAL': '''
Your task is to generate a Python dictionary that shows the named entities in the given text. Your response should be just a Python dictionary where the type of each named entity is the key and the entities of each type are values of that key.

Entity types and their description are as follows:
Person: Names of people.
PersonType: Job types or roles held by a person.
Location: Natural and human-made landmarks, structures, geographical features, and geopolitical entities.
Organization: Companies, political groups, musical bands, sports clubs, government bodies, and public organizations. Nationalities and religions are not included in this entity type.
Event: Historical, social, and naturally occurring events (like Cultural, Natural, and Sports events).
Product: Physical objects of various categories.
Skill: A capability, skill, or expertise.
PhoneNumber: Phone numbers.
Email: Email addresses.
URL: URLs to websites.
IP: Network IP addresses.
DateTime: Dates and times of day (like Date, Time, DateRange, TimeRange, Duration, and Set).
Quantity: Numbers and numeric quantities (like Number, Percentage, Age, Currency, Dimensions, and Temperature).

Here is one example:
Input: Nikon 55-200mm 0.28x - 55mm 200mm - f/4 5.6
Output: 
{ "Organization": ["Nikon"], "Quantity": ["55-200mm", "0.28x", "55mm", "200mm", "f/4 5.6"] }
Input: 
''',
        'NER_DIC_FIXED_QUERRY': '''\nOutput: ?''',
    }
)

# Address: Mailing address.

MAPPING_PROMPTS = namedtuple('MAPPING_HANDLER_PROMPTS', 'VALUE')
MAPPING_HANDLER_PROMPTS = MAPPING_PROMPTS(
    VALUE={
        'MAPPING_HANDLER_INITIAL':
        '''
Act as a function that gets a <src> value and returns a <trg> value (<rule>). Your answer should be just <trg> name without any other description. You will be given a few examples to guide your response.
Here are examples:
''',
        'MAPPING_HANDLER_FIXED_QUERRY': ''' <trg>: ?''',
    })
