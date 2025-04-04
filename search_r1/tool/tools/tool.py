from typing import List, Dict, Any, Optional, Tuple
from search_r1.tool.tools.attractions.apis import Attractions
from search_r1.tool.tools.restaurants.apis import Restaurants
from search_r1.tool.tools.flights.apis import Flights
from search_r1.tool.tools.accommodations.apis import Accommodations
from search_r1.tool.tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from search_r1.tool.tools.cities.apis import Cities
import pandas as pd

class ToolError(Exception):
    """搜索错误的自定义异常类"""
    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": False,
            "data": [],
            "error_type": self.error_type,
            "message": self.message,
            "total": 0
        }

class Tool:
    def __init__(self) -> None:
        """初始化所有API服务"""
        try:
            self.attractions = Attractions()
            self.restaurants = Restaurants()
            self.flights = Flights()
            self.accommodations = Accommodations()
            self.distance = GoogleDistanceMatrix()
            self.cities = Cities()
            print("搜索API初始化完成。")
        except Exception as e:
            raise ToolError("初始化错误", f"API服务初始化失败：{str(e)}")

    def search_attractions(self, 
                         city: Optional[str] = None) -> Dict[str, Any]:
        """搜索景点
        
        Args:
            city: 所在城市
        """
        try:
            results = self.attractions.run(city=city)
            if isinstance(results, str):  # 处理错误消息
                return {
                    "success": False,
                    "data": [],
                    "error_type": "attraction_error",
                    "message": results,
                    "total": 0
                }
            return {
                "success": True,
                "data": results.to_dict('records'),
                "error_type": None,
                "message": "搜索成功" if not results.empty else "未找到符合条件的景点",
                "total": len(results)
            }
        except Exception as e:
            error_msg = {
                "景点不存在": "未找到符合条件的景点",
                "参数错误": "搜索参数有误，请检查输入",
                "服务异常": "景点搜索服务暂时不可用"
            }.get(str(e), f"景点搜索失败：{str(e)}")
            return {
                "success": False,
                "data": [],
                "error_type": "attraction_error",
                "message": error_msg,
                "total": 0
            }

    def search_restaurants(self,
                         city: Optional[str] = None) -> Dict[str, Any]:
        """搜索餐厅
        
        Args:
            city: 所在城市
        """
        try:
            results = self.restaurants.run(city=city)
            if isinstance(results, str):  # 处理错误消息
                return {
                    "success": False,
                    "data": [],
                    "error_type": "restaurant_error",
                    "message": results,
                    "total": 0
                }
            return {
                "success": True,
                "data": results.to_dict('records'),
                "error_type": None,
                "message": "搜索成功" if not results.empty else "未找到符合条件的餐厅",
                "total": len(results)
            }
        except Exception as e:
            error_msg = {
                "餐厅不存在": "未找到符合条件的餐厅",
                "参数错误": "搜索参数有误，请检查输入",
                "服务异常": "餐厅搜索服务暂时不可用"
            }.get(str(e), f"餐厅搜索失败：{str(e)}")
            return {
                "success": False,
                "data": [],
                "error_type": "restaurant_error",
                "message": error_msg,
                "total": 0
            }

    def search_flights(self,
                      origin_city: str,
                      dest_city: str,
                      date: str) -> Dict[str, Any]:
        """搜索航班
        
        Args:
            origin_city: 出发城市
            dest_city: 目的城市
            date: 出发日期（YYYY-MM-DD格式）
        """
        try:
            results = self.flights.run(
                origin=origin_city,
                destination=dest_city,
                departure_date=date
            )
            if isinstance(results, str):  # 处理错误消息
                return {
                    "success": False,
                    "data": [],
                    "error_type": "flight_error",
                    "message": results,
                    "total": 0
                }
            return {
                "success": True,
                "data": results.to_dict('records'),
                "error_type": None,
                "message": "搜索成功" if not results.empty else "未找到符合条件的航班",
                "total": len(results)
            }
        except Exception as e:
            error_msg = {
                "航班不存在": "未找到符合条件的航班",
                "参数错误": "搜索参数有误，请检查输入",
                "日期格式错误": "日期格式错误，请使用YYYY-MM-DD格式",
                "服务异常": "航班搜索服务暂时不可用"
            }.get(str(e), f"航班搜索失败：{str(e)}")
            return {
                "success": False,
                "data": [],
                "error_type": "flight_error",
                "message": error_msg,
                "total": 0
            }

    def search_accommodations(self,
                            city: str) -> Dict[str, Any]:
        """搜索住宿
        
        Args:
            city: 城市名称
        """
        try:
            results = self.accommodations.run(city=city)
            if isinstance(results, str):  # 处理错误消息
                return {
                    "success": False,
                    "data": [],
                    "error_type": "accommodation_error",
                    "message": results,
                    "total": 0
                }
            return {
                "success": True,
                "data": results.to_dict('records'),
                "error_type": None,
                "message": "搜索成功" if not results.empty else "未找到符合条件的住宿",
                "total": len(results)
            }
        except Exception as e:
            error_msg = {
                "住宿不存在": "未找到符合条件的住宿",
                "参数错误": "搜索参数有误，请检查输入",
                "服务异常": "住宿搜索服务暂时不可用"
            }.get(str(e), f"住宿搜索失败：{str(e)}")
            return {
                "success": False,
                "data": [],
                "error_type": "accommodation_error",
                "message": error_msg,
                "total": 0
            }

    def search_cities(self,
                      state: str) -> Dict[str, Any]:
        """搜索城市
        
        Args:
            state: 州名称
        """
        try:
            results = self.cities.run(state=state)
            if isinstance(results, str):  # 处理错误消息
                return {
                    "success": False,
                    "data": [],
                    "error_type": "city_error",
                    "message": results,
                    "total": 0
                }
            return {
                "success": True,
                "data": results,
                "error_type": None,
                "message": "搜索成功",
                "total": len(results)
            }
        except Exception as e:
            error_msg = {
                "参数错误": "搜索参数有误，请检查输入",
                "服务异常": "城市搜索服务暂时不可用"
            }.get(str(e), f"城市搜索失败：{str(e)}")
            return {
                "success": False,
                "data": [],
                "error_type": "city_error",
                "message": error_msg,
                "total": 0
            }   

    def calculate_distance(self, 
                         origin: str, 
                         destination: str, 
                         mode: str = "driving") -> Dict[str, Any]:
        """计算两地之间的距离和时间
        
        Args:
            origin: 出发地点
            destination: 目的地点
            mode: 交通方式，可选值：['driving', 'taxi', 'walking', 'transit']
        """
        try:
            results = self.distance.run(
                origin=origin,
                destination=destination,
                mode=mode
            )
            if isinstance(results, str) and "no valid information" in results.lower():
                return {
                    "success": False,
                    "data": [],
                    "error_type": "distance_error",
                    "message": "无法计算指定地点间的距离",
                    "total": 0
                }
            return {
                "success": True,
                "data": [{"result": results}],
                "error_type": None,
                "message": "距离计算成功",
                "total": 1
            }
        except Exception as e:
            error_msg = {
                "参数错误": "计算参数有误，请检查输入",
                "无效地点": "提供的地点无效",
                "服务异常": "距离计算服务暂时不可用"
            }.get(str(e), f"距离计算失败：{str(e)}")
            return {
                "success": False,
                "data": [],
                "error_type": "distance_error",
                "message": error_msg,
                "total": 0
            }