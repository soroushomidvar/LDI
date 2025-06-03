from collections import namedtuple

FIXED_ONE_SHOT_TEXT = 'You are given one example to guide your response: '
FIXED_FEW_SHOT_TEXT = 'You are given some examples to guide your response: '

DATASET_PROMPTS = namedtuple('DATASET_PROMPTS', 'VALUE')
MAPPING_PROMPTS = namedtuple('MAPPING_HANDLER_PROMPTS', 'VALUE') #Act as a function that gets <src> value and returns a <trg> value (<rule>). 
MAPPING_HANDLER_PROMPTS = MAPPING_PROMPTS(
    VALUE={
        'MAPPING_HANDLER_INITIAL':
'''
Your answer should be just <trg> name without any other description. You will be given a few examples to guide your response. Here are examples:
''',
        'MAPPING_HANDLER_MIDDLE':
'''Now return the <trg> without any other description:
''',
        'MAPPING_HANDLER_QUERRY': '''<trg>: ? '''
    })
