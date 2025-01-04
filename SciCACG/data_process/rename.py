import json

json_file = "../data/new_compare.jsonl"  # JSON文件路径
data_list = []
# 读取JSON文件
with open(json_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        data_list.append(data)
new_data = []
# 遍历JSON数据
for obj in data_list:
    # 根据条件划分为新的值
    if obj['Is_compare'] == '1':
        obj['Is_compare'] = 'comparable'
    else:
        obj['Is_compare'] = 'Nocomparable'

    new_obj = {
        'citing_paper_id': obj['citing_paper_id'],
        'cited_paper_id': obj['cited_paper_id'],
        'citing_paper_abstract': obj['citing_paper_abstract'],
        'cited_paper_abstract': obj['cited_paper_abstract'],
        'citation': obj['citation'],
        'Is_compare': obj['Is_compare']
    }

    new_data.append(new_obj)

with open('../data/new_ComparePaper.jsonl', 'w', encoding='utf-8') as f:
    for obj in new_data:
        json.dump(obj, f)
        f.write('\n')

print("操作完成，源文件已被修改")