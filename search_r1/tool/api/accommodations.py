import json
import os
from typing import List, Dict, Any

class AccommodationAPI:
    def __init__(self):
        self.accommodations_data = []
        self._load_data()
    
    def _load_data(self):
        """加载住宿数据"""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_dir = os.path.join(current_dir, "database")
        file_path = os.path.join(json_dir, "accommodations.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.accommodations_data = json.load(f)
        except FileNotFoundError:
            print(f"警告: 未找到住宿数据文件 {file_path}")
            self.accommodations_data = []
    
    def search_accommodations(self,
                            city: str = None,
                            room_type: str = None,
                            house_rules: str = None,
                            people_number: int = None,
                            price_range: tuple[float, float] = None,
                            min_rating: float = None) -> List[Dict[str, Any]]:
        """
        搜索住宿信息
        
        Args:
            city: 城市名称
            room_type: 房间类型
            house_rules: 住宿规则
            people_number: 入住人数
            price_range: 价格范围
            min_rating: 最低评分
            
        Returns:
            符合条件的住宿列表
        """
        results = self.accommodations_data
        
        if city:
            results = [a for a in results if a.get("City", "").lower() == city.lower()]
                
        if room_type:
            results = [a for a in results if room_type.lower() in a.get("room type", "").lower()]

        if house_rules:
            results = [a for a in results if house_rules.lower() in a.get("house_rules", "").lower()]
        
        if people_number is not None:
            results = [a for a in results if a.get("maximum occupancy", 0) >= people_number]
        
        if price_range is not None:
            results = [a for a in results if a.get("price", 0) >= price_range[0] and a.get("price", 0) <= price_range[1]]
        
        if min_rating is not None:
            results = [a for a in results if a.get("review rate number", 0) >= min_rating]
        
        return results
    
    def get_accommodation_by_id(self, accommodation_id: str) -> Dict[str, Any]:
        """
        根据ID获取住宿信息
        
        Args:
            accommodation_id: 住宿ID
            
        Returns:
            住宿信息字典
        """
        for accommodation in self.accommodations_data:
            if str(accommodation.get("ID", "")) == str(accommodation_id):
                return accommodation
        return {}
    
    def get_accommodation_types(self) -> List[str]:
        """
        获取所有住宿类型列表
        
        Returns:
            住宿类型列表
        """
        types = set()
        for accommodation in self.accommodations_data:
            if accommodation.get("Type"):
                types.update(type.strip() for type in accommodation["Type"].split(","))
        return sorted(list(types))
    