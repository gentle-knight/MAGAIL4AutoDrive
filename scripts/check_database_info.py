import pickle
import os

# 检查过滤后的数据库
filtered_db = "/home/huangfukk/mdsn/exp_filtered"

print("="*60)
print("过滤后数据库信息")
print("="*60)

# 读取summary
summary_path = os.path.join(filtered_db, "dataset_summary.pkl")
with open(summary_path, 'rb') as f:
    summary = pickle.load(f)

print(f"\n总场景数: {len(summary)}")
print(f"场景ID列表(前10个): {list(summary.keys())[:10]}")

# 读取mapping
mapping_path = os.path.join(filtered_db, "dataset_mapping.pkl")
with open(mapping_path, 'rb') as f:
    mapping = pickle.load(f)

print(f"\n映射关系数量: {len(mapping)}")

# 检查第一个场景的详细信息
first_scenario_id = list(summary.keys())[0]
first_scenario_info = summary[first_scenario_id]
print(f"\n第一个场景详细信息:")
print(f"  场景ID: {first_scenario_id}")
print(f"  元数据: {first_scenario_info}")

# 检查映射的文件路径
first_scenario_path = mapping[first_scenario_id]
print(f"  场景文件路径(相对): {first_scenario_path}")

# 检查文件是否存在
abs_path = os.path.join(filtered_db, first_scenario_path)
print(f"  场景文件路径(绝对): {abs_path}")
print(f"  文件存在: {os.path.exists(abs_path)}")

# 统计源数据库的场景文件
converted_db = "/home/huangfukk/mdsn/exp_converted"
converted_files = [f for f in os.listdir(converted_db) if f.endswith('.pkl') and f.startswith('sd_')]
print(f"\n源数据库 exp_converted:")
print(f"  场景文件数量: {len(converted_files)}")
print(f"  示例文件: {converted_files[:5]}")
