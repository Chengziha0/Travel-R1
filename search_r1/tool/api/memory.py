import pandas as pd
import json

class Memory:
    def __init__(self):
        """初始化空的内存字典"""
        self.data = {}

    def write(self, key: str, value) -> str:
        """写入或更新数据"""
        self.data[key] = value
        return f"已保存数据，键为 '{key}'"

    def read(self, key: str):
        """读取数据"""
        if key in self.data:
            return self.data[key]
        else:
            return f"错误：键 '{key}' 不存在"

    def delete(self, key: str) -> str:
        """删除指定键的数据"""
        if key in self.data:
            del self.data[key]
            return f"已删除键 '{key}' 的数据"
        else:
            return f"错误：键 '{key}' 不存在，无法删除"

    def list_keys(self) -> list:
        """列出所有键"""
        return list(self.data.keys())

    def list_all(self) -> list:
        """列出所有键和值的摘要"""
        results = []
        for key, value in self.data.items():
            # 根据数据类型生成摘要
            if isinstance(value, pd.DataFrame):
                summary = value.head().to_string(index=False)
            elif isinstance(value, (list, dict)):
                summary = json.dumps(value, ensure_ascii=False, indent=2)
            else:
                summary = str(value)
            results.append({"key": key, "summary": summary})
        return results

    def reset(self):
        """清空所有数据"""
        self.data = {}
        return "内存已清空"