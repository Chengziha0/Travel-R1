import json
import os
from typing import List, Dict, Any

class AttractionAPI:
    def __init__(self):
        self.attractions_data = []
        self._load_data()
    
    def _load_data(self):
        """加载景点数据"""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_dir = os.path.join(current_dir, "database")
        file_path = os.path.join(json_dir, "attractions.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.attractions_data = json.load(f)
        except FileNotFoundError:
            print(f"警告: 未找到景点数据文件 {file_path}")
            self.attractions_data = []
    
    def search_attractions(self,
                         city: str = None,
                         name: str = None,) -> List[Dict[str, Any]]:
        """
        搜索景点信息
        
        Args:
            city: 城市名称
            name: 景点名称
            
        Returns:
            符合条件的景点列表
        """
        results = self.attractions_data
        
        if city:
            results = [a for a in results if a.get("City", "").lower() == city.lower()]
        
        if name:
            results = [a for a in results if name.lower() in a.get("Name", "").lower()]
        
        return results
    
    def get_attraction_by_id(self, attraction_id: str) -> Dict[str, Any]:
        """
        根据ID获取景点信息
        
        Args:
            attraction_id: 景点ID
            
        Returns:
            景点信息字典
        """
        for attraction in self.attractions_data:
            if str(attraction.get("ID", "")) == str(attraction_id):
                return attraction
        return {}
    
    def get_cities(self) -> List[str]:
        """
        获取所有城市列表
        
        Returns:
            城市名称列表
        """
        return sorted(list(set(a.get("City", "") for a in self.attractions_data if a.get("City")))) 