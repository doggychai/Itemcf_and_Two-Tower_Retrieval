# import json
#
# # 文件路径
# file_path = "/ExtHDD/Users/chai/recommend_news/Industrial_and_Scientific.item.json"
#
# # 读取并解析 JSON 文件
# try:
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     # 打印文件内容
#     print("=" * 50)
#     print("文件内容预览:")
#     print("=" * 50)
#
#     # 格式化输出 JSON
#     print(json.dumps(data, indent=2, ensure_ascii=False))
#
#     # 如果是列表，显示基本信息
#     if isinstance(data, list):
#         print(f"\n总条目数: {len(data)}")
#         if len(data) > 0:
#             print(f"第一条数据类型: {type(data[0])}")
#             if isinstance(data[0], dict):
#                 print(f"第一条数据的键: {list(data[0].keys())}")
#
#     # 如果是字典，显示键
#     elif isinstance(data, dict):
#         print(f"\n顶层键: {list(data.keys())}")
#
# except FileNotFoundError:
#     print(f"错误: 文件未找到 - {file_path}")
# except json.JSONDecodeError as e:
#     print(f"错误: JSON 解析失败 - {e}")
# except Exception as e:
#     print(f"错误: {e}")
nums = [-4,-1,-1,0,5,5]

a = sorted(nums)
print(a)