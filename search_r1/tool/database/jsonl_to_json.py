import json
import os

def convert_jsonl_to_json(input_file):
    """将JSONL文件转换为多个JSON文件，根据数据中的类别分类"""
    # 初始化数据存储字典
    categories = {}
    flights = []
    
    # 创建flights文件夹
    output_dir = "json_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取JSONL文件
    print("正在读取JSONL文件...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # 遍历每个键值对
                for category, items in data.items():
                    # 如果是航班相关的数据
                    if "Flight" in category and isinstance(items, list):
                        # 为每个航班添加路线信息
                        for flight in items:
                            flight["route"] = category
                            flights.append(flight)
                    # 如果是其他列表类型的数据
                    elif isinstance(items, list):
                        # 提取类别名称（去掉城市名称）
                        pure_category = category.split(" in ")[0] if " in " in category else category
                        if pure_category not in categories:
                            categories[pure_category] = []
                        categories[pure_category].extend(items)
                    # 如果是其他类型的数据（比如字符串），也可以保存
                    elif items != "No valid information." and not isinstance(items, str):
                        if category not in categories:
                            categories[category] = []
                        categories[category].append(items)
            except json.JSONDecodeError:
                print(f"警告: 跳过无效的JSON行")
                continue

    # 将每个类别的数据保存为单独的JSON文件
    for category, items in categories.items():
        if items:
            # 将类别名称中的空格替换为下划线，并转换为小写
            filename = f"{category.lower().replace(' ', '_')}.json"
            filepath = os.path.join(output_dir, filename)
                
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"已生成 {filepath}")
    
    # 保存所有航班数据
    if flights:
        filepath = os.path.join(output_dir, "flights.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(flights, f, ensure_ascii=False, indent=2)
        print(f"已生成 {filepath}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "train_ref_info.jsonl")
    convert_jsonl_to_json(input_file) 