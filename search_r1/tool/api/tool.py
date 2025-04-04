from typing import List, Dict, Any, Optional
from search_r1.tool.api.attractions import AttractionAPI
from search_r1.tool.api.restaurants import RestaurantAPI
from search_r1.tool.api.flights import FlightAPI
from search_r1.tool.api.accommodations import AccommodationAPI
# from search_r1.tool.api.distance import GoogleDistanceMatrix

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
            self.attractions = AttractionAPI()
            self.restaurants = RestaurantAPI()
            self.flights = FlightAPI()
            self.accommodations = AccommodationAPI()
            # self.distance = GoogleDistanceMatrix()
            print("搜索API初始化完成。")
        except Exception as e:
            raise ToolError("初始化错误", f"API服务初始化失败：{str(e)}")

    def search_attractions(self, 
                         name: Optional[str] = None,
                         city: Optional[str] = None) -> Dict[str, Any]:
        """搜索景点
        
        Args:
            name: 景点名称关键词
            city: 所在城市
        """
        try:
            results = self.attractions.search_attractions(
                name=name,
                city=city,
            )
            return {
                "success": True,
                "data": results,
                "error_type": None,
                "message": "搜索成功" if results else "未找到符合条件的景点",
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
                         city: Optional[str] = None,
                         cuisine: Optional[str] = None,) -> Dict[str, Any]:
        """搜索餐厅
        
        Args:
            city: 所在城市
            cuisine: 菜系类型
        """
        try:
            results = self.restaurants.search_restaurants(
                city=city,
                cuisine=cuisine,
            )
            return {
                "success": True,
                "data": results,
                "error_type": None,
                "message": "搜索成功" if results else "未找到符合条件的餐厅",
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
                      origin_city: Optional[str] = None,
                      dest_city: Optional[str] = None,
                      date: Optional[str] = None,
                      deptime_time: Optional[tuple[str, str]] = None,
                      arrtime_time: Optional[tuple[str, str]] = None) -> List[Dict[str, Any]]:
        """搜索航班
        
        Args:
            origin_city: 出发城市
            dest_city: 目的城市
            date: 日期（YYYY-MM-DD格式）
            deptime_time: 出发时间区间
            arrtime_time: 到达时间区间
        """
        try:
            results = self.flights.search_flights(
                origin_city=origin_city,
                dest_city=dest_city,
                date=date,
                deptime_time=deptime_time,
                arrtime_time=arrtime_time
            )
            return {
                "success": True,
                "data": results,
                "error_type": None,
                "message": "搜索成功" if results else "未找到符合条件的航班",
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
                            city: str = None,
                            name: str = None,
                            room_type: str = None,
                            house_rules: str = None,
                            people_number: int = None,
                            price_range: tuple[float, float] = None,
                            min_rating: float = None) -> List[Dict[str, Any]]:
        """搜索住宿
        
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
        try:
            results = self.accommodations.search_accommodations(
                city=city,
                room_type=room_type,
                house_rules=house_rules,
                people_number=people_number,
                price_range=price_range,
                min_rating=min_rating
            )
            return {
                "success": True,
                "data": results,
                "error_type": None,
                "message": "搜索成功" if results else "未找到符合条件的住宿",
                "total": len(results)
            }
        except Exception as e:
            error_msg = {
                "住宿不存在": "未找到符合条件的住宿",
                "参数错误": "搜索参数有误，请检查输入",
                "评分范围错误": "评分必须在0-5之间",
                "服务异常": "住宿搜索服务暂时不可用"
            }.get(str(e), f"住宿搜索失败：{str(e)}")
            return {
                "success": False,
                "data": [],
                "error_type": "accommodation_error",
                "message": error_msg,
                "total": 0
            }

    # def distance(self, origin: str, destination: str, mode: str = "driving") -> Dict[str, Any]:
    #     """计算距离
        
    #     Args:
    #         origin: 出发地
    #         destination: 目的地
    #         mode: 交通方式, options: ['driving', 'taxi', 'walking', 'distance', 'transit']
            
    #     Returns:
    #         距离信息
    #     """
    #     try:
    #         results = self.distance.run_online(origin, destination, mode)
    #         return {
    #             "success": True,
    #             "data": results,
    #             "error_type": None,
    #             "message": "距离计算成功",
    #             "total": 1
    #         }           
    #     except Exception as e:
    #         error_msg = {
    #             "距离计算失败": "距离计算失败",
    #             "参数错误": "计算参数有误，请检查输入",
    #             "服务异常": "距离计算服务暂时不可用"
    #         }.get(str(e), f"距离计算失败：{str(e)}")
    #         return {
    #             "success": False,
    #             "data": [],
    #             "error_type": "distance_error",
    #             "message": error_msg,
    #             "total": 0
    #         }
        
    