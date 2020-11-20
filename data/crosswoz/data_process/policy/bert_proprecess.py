import os
import json

import torch

from transformers import BertTokenizer, BertConfig

DOMAINS = ['餐馆', '景点', '酒店', '地铁', '出租', 'reqmore']
INTENT = ['Inform', 'Recommend', 'NoOffer', 'General']
SLOTS = ['名称', '门票', '游玩时间', '评分', '周边景点', '周边餐馆', '周边酒店',
         '推荐菜', '人均消费', '酒店类型', '酒店设施', '价格', '出发地', '目的地']

