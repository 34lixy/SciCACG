import json
from collections import defaultdict

json_file = "../data/new_ComparePaper.jsonl"  # JSON文件路径
property_name = "Is_compare"  # 属性名称

# 读取JSON文件并将数据存储在列表中
data_list = []
with open(json_file, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        data_list.append(data)

# 统计分类数量
category_counts = defaultdict(int)
for item in data_list:
    category = item.get(property_name)
    if category is not None:
        category_counts[category] += 1

# 输出分类及其数量
for category, count in category_counts.items():
    print(f"{category}: {count}")
