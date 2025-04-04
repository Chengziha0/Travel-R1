import json
import os
from typing import List, Dict, Any
from math import radians, sin, cos, sqrt, atan2

class RestaurantAPI:
    def __init__(self):
        self.restaurants_data = []
        self._load_data()
    
    def _load_data(self):
        """加载餐厅数据"""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_dir = os.path.join(current_dir, "database")
        file_path = os.path.join(json_dir, "restaurants.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.restaurants_data = json.load(f)
        except FileNotFoundError:
            print(f"警告: 未找到餐厅数据文件 {file_path}")
            self.restaurants_data = []
    
    def search_restaurants(self,
                        city: str = None,
                        cuisine: str = None,) -> List[Dict[str, Any]]:
        """
        搜索餐厅信息
        
        Args:
            city: 城市名称
            cuisine: 菜系
            
        Returns:
            符合条件的餐厅列表
        """
        results = self.restaurants_data
        
        if city:
            results = [r for r in results if r.get("City", "").lower() == city.lower()]
        
        if cuisine:
            results = [r for r in results if cuisine.lower() in r.get("Cuisine", "").lower()]
        
        return results
    
    def get_restaurant_by_id(self, restaurant_id: str) -> Dict[str, Any]:
        """
        根据ID获取餐厅信息
        
        Args:
            restaurant_id: 餐厅ID
            
        Returns:
            餐厅信息字典
        """
        for restaurant in self.restaurants_data:
            if str(restaurant.get("ID", "")) == str(restaurant_id):
                return restaurant
        return {}
    
    def get_cuisines(self) -> List[str]:
        """
        获取所有菜系列表
        
        Returns:
            菜系名称列表
        """
        cuisines = set()
        for restaurant in self.restaurants_data:
            if restaurant.get("Cuisine"):
                cuisines.update(cuisine.strip() for cuisine in restaurant["Cuisine"].split(","))
        return sorted(list(cuisines))

class Restaurants:
    def __init__(self, path=None) -> None:
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.path = os.path.join(current_dir, "..", "database", "train_ref_info.jsonl")
        else:
            self.path = path
        self.data = {}
        self.load_data()
        print("Restaurants loaded.")

    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    for key, value in data.items():
                        if "Restaurants" in key:
                            self.data[key] = value

    def search_by_city(self, city: str) -> List[Dict[str, Any]]:
        """根据城市名称搜索餐厅"""
        key = f"Restaurants in {city}"
        return self.data.get(key, [])

    def search_by_name(self, name: str) -> List[Dict[str, Any]]:
        """根据餐厅名称搜索"""
        results = []
        for restaurants in self.data.values():
            for restaurant in restaurants:
                if name.lower() in restaurant["Name"].lower():
                    results.append(restaurant)
        return results

    def search_by_cuisine(self, cuisine: str) -> List[Dict[str, Any]]:
        """根据菜系搜索餐厅"""
        results = []
        for restaurants in self.data.values():
            for restaurant in restaurants:
                if "Cuisine" in restaurant and cuisine.lower() in restaurant["Cuisine"].lower():
                    results.append(restaurant)
        return results

    def get_nearby_restaurants(self, lat: float, lon: float, radius: float = 5) -> List[Dict[str, Any]]:
        """根据经纬度搜索附近餐厅"""
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # 地球半径（公里）
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c

        results = []
        for restaurants in self.data.values():
            for restaurant in restaurants:
                if "Latitude" in restaurant and "Longitude" in restaurant:
                    dist = haversine_distance(
                        lat, lon,
                        restaurant["Latitude"],
                        restaurant["Longitude"]
                    )
                    if dist <= radius:
                        restaurant["Distance"] = round(dist, 2)
                        results.append(restaurant)
        
        return sorted(results, key=lambda x: x["Distance"])
