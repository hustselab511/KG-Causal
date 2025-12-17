import json
import random
from collections import OrderedDict
import numpy as np
import torch
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

def format_dict_to_json(data_dict, indent=4, sort_keys=False):
    """将Python字典格式化为带缩进的JSON字符串

    Args:
        data_dict: 原始Python字典对象
        indent: 缩进空格数，默认4个空格
        sort_keys: 是否按字母顺序排序键，默认False（保持原始顺序）

    Returns:
        格式化后的JSON字符串
    """
    try:
        # 直接将字典转换为格式化的JSON字符串
        formatted_json = json.dumps(
            data_dict,
            ensure_ascii=False,  # 保留中文等非ASCII字符
            indent=indent,  # 设置缩进
            sort_keys=sort_keys  # 是否排序键
        )
        return formatted_json
    except TypeError as e:
        return f"字典序列化错误: {str(e)}（可能包含非JSON可序列化对象）"

def load_json_args(path):
    json_str = ''
    with open(path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'  #
            json_str += line
    defaults = json.loads(json_str, object_pairs_hook=OrderedDict)
    dict_args = {}
    for key in defaults.keys():
        dict_args.update(defaults[key])
    return dict_args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_strings(strings: List[str], sort_by: str = 'count', descending: bool = True) -> Tuple[
    Dict[str, int], int, int]:
    """
    统计字符串列表中各字符串的出现次数
    :param strings: 输入字符串列表
    :param sort_by: 排序方式，可选 'count'（按次数）或 'string'（按字符串）
    :param descending: 是否降序排列
    :return: (统计结果字典, 总字符串数, 字符串种类数)
    """
    if not isinstance(strings, list) or not all(isinstance(s, str) for s in strings):
        raise ValueError("输入必须是字符串列表")

    # 使用Counter统计（高效且简洁）
    count_result = Counter(strings)

    # 按需求排序
    if sort_by == 'count':
        sorted_items = sorted(count_result.items(), key=lambda x: x[1], reverse=descending)
    elif sort_by == 'string':
        sorted_items = sorted(count_result.items(), key=lambda x: x[0], reverse=descending)
    else:
        raise ValueError("sort_by参数必须是'count'或'string'")

    # 转换为有序字典（保持排序结果）
    sorted_result = {k: v for k, v in sorted_items}

    return sorted_result, len(strings), len(count_result)