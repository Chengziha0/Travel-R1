import json
import os
from typing import List, Dict, Any
from datetime import datetime

class FlightAPI:
    def __init__(self):
        self.flights_data = []
        self._load_data()
    
    def _load_data(self):
        """加载航班数据"""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_dir = os.path.join(current_dir, "database")
        file_path = os.path.join(json_dir, "flights.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.flights_data = json.load(f)
        except FileNotFoundError:
            print(f"警告: 未找到航班数据文件 {file_path}")
            self.flights_data = []
    
    def search_flights(self, 
                      origin_city: str = None,
                      dest_city: str = None,
                      date: str = None,
                      deptime_time: tuple[str, str] = None,
                      arrtime_time: tuple[str, str] = None) -> List[Dict[str, Any]]:
        """
        搜索航班信息
        
        Args:
            origin_city: 出发城市
            dest_city: 目的地城市
            date: 航班日期
            deptime_time: 出发时间区间
            arrtime_time: 到达时间区间
            
        Returns:
            符合条件的航班列表
        """
        results = self.flights_data
        
        if origin_city:
            results = [f for f in results if f.get("OriginCityName", "").lower() == origin_city.lower()]
        
        if dest_city:
            results = [f for f in results if f.get("DestCityName", "").lower() == dest_city.lower()]
        
        if date:
            results = [f for f in results if f.get("FlightDate", "") == date]
        
        if deptime_time is not None:
            results = [f for f in results if f.get("DepTime", "") >= deptime_time[0] and f.get("DepTime", "") <= deptime_time[1]]
        
        if arrtime_time is not None:
            results = [f for f in results if f.get("ArrTime", "") >= arrtime_time[0] and f.get("ArrTime", "") <= arrtime_time[1]]
        
        return results
    
    def get_flight_by_number(self, flight_number: str) -> Dict[str, Any]:
        """
        根据航班号获取航班信息
        
        Args:
            flight_number: 航班号
            
        Returns:
            航班信息字典
        """
        for flight in self.flights_data:
            if flight.get("Flight Number", "").lower() == flight_number.lower():
                return flight
        return {}
    
    def get_available_dates(self, origin_city: str = None, dest_city: str = None) -> List[str]:
        """
        获取可用的航班日期
        
        Args:
            origin_city: 出发城市
            dest_city: 目的地城市
            
        Returns:
            可用日期列表
        """
        results = self.flights_data
        
        if origin_city:
            results = [f for f in results if f.get("OriginCityName", "").lower() == origin_city.lower()]
        
        if dest_city:
            results = [f for f in results if f.get("DestCityName", "").lower() == dest_city.lower()]
        
        return sorted(list(set(f.get("FlightDate", "") for f in results if f.get("FlightDate"))))

class Flights:
    def __init__(self, path=None) -> None:
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.path = os.path.join(current_dir, "..", "database", "train_ref_info.jsonl")
        else:
            self.path = path
        self.data = {}
        self.load_data()
        print("Flights loaded.")

    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    for key, value in data.items():
                        if "Flights" in key:
                            self.data[key] = value

    def search_by_city(self, city: str, is_departure: bool = True) -> List[Dict[str, Any]]:
        """根据城市搜索航班"""
        results = []
        city_key = "DepartureCity" if is_departure else "ArrivalCity"
        for flights in self.data.values():
            for flight in flights:
                if city_key in flight and city.lower() in flight[city_key].lower():
                    results.append(flight)
        return results

    def search_by_route(self, departure: str, arrival: str) -> List[Dict[str, Any]]:
        """根据出发地和目的地搜索航班"""
        results = []
        for flights in self.data.values():
            for flight in flights:
                if ("DepartureCity" in flight and "ArrivalCity" in flight and
                    departure.lower() in flight["DepartureCity"].lower() and
                    arrival.lower() in flight["ArrivalCity"].lower()):
                    results.append(flight)
        return results

    def search_by_date(self, date: str, departure: str = "", arrival: str = "") -> List[Dict[str, Any]]:
        """根据日期搜索航班，可选择指定出发地和目的地"""
        try:
            search_date = datetime.strptime(date, "%Y-%m-%d")
            results = []
            for flights in self.data.values():
                for flight in flights:
                    if "Date" in flight:
                        flight_date = datetime.strptime(flight["Date"], "%Y-%m-%d")
                        if (flight_date.date() == search_date.date() and
                            (not departure or departure.lower() in flight.get("DepartureCity", "").lower()) and
                            (not arrival or arrival.lower() in flight.get("ArrivalCity", "").lower())):
                            results.append(flight)
            return results
        except ValueError:
            return []  # 日期格式错误

    def search_by_flight_number(self, flight_number: str) -> Dict[str, Any]:
        """根据航班号搜索航班"""
        for flights in self.data.values():
            for flight in flights:
                if "FlightNumber" in flight and flight_number.lower() == flight["FlightNumber"].lower():
                    return flight
        return {}