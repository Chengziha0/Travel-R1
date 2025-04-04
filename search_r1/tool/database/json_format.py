import json
import os

def analyze_first_record(file_path):
    """只分析JSONL文件中的第一条数据的结构"""
    print("\n分析第一条数据的结构...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 只读取第一行
        first_line = f.readline().strip()
        if first_line:
            try:
                data = json.loads(first_line)
                print("\n数据类型及其字段:")
                for key, value in data.items():
                    # 如果是列表类型的数据
                    if isinstance(value, list) and value:
                        # 获取第一个元素的所有字段名
                        fields = sorted(value[0].keys()) if value else []
                        print(f"\n{key}:")
                        for field in fields:
                            print(f"  - {field}")
            except json.JSONDecodeError:
                print("错误: 无效的JSON格式")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "train_ref_info.jsonl")
    analyze_first_record(file_path)